'''Twix "meas" file handling

Inspired by and includes code from "vespa" (http://scion.duhs.duke.edu/vespa/)
'''

import os, struct, sys, re
import pickle
from datetime import datetime
from collections import namedtuple, deque, OrderedDict
from itertools import product as iproduct
from copy import deepcopy
from functools import reduce
from distutils.version import LooseVersion # for syngo version comparision

import numpy as np


def _read_cstr(source_file):
    '''Read a null terminated byte string'''
    chars = [ ]
    char = source_file.read(1)
    while char != '\x00':
        chars.append(char)
        char = source_file.read(1)

    return ''.join(chars)


class HeaderElemSpec(object):
    '''Describes a single element in a structured binary header'''
    def __init__(self, struct_char, n_elems=1):
        self.struct_char = struct_char
        self.n_elems = n_elems

    def get_struct_fmt_str(self):
        if self.n_elems == 1:
            return self.struct_char
        else:
            return "%d%s" % (self.n_elems, self.struct_char)


mdh_elem_specs = {'dma_info' : HeaderElemSpec('I'),
                  'meas_uid' : HeaderElemSpec('i'),
                  'scan_count' : HeaderElemSpec('I'),
                  'timestamp' : HeaderElemSpec('I'),
                  'pmu_timestamp' : HeaderElemSpec('I'),
                  'eval_info_mask' : HeaderElemSpec('Q'),
                  'samples_in_scan' : HeaderElemSpec('H'),
                  'used_channels' : HeaderElemSpec('H'),
                  'line' : HeaderElemSpec('H'),
                  'acquisition' : HeaderElemSpec('H'),
                  'slice' : HeaderElemSpec('H'),
                  'partition' : HeaderElemSpec('H'),
                  'echo' : HeaderElemSpec('H'),
                  'phase' : HeaderElemSpec('H'),
                  'repetition' : HeaderElemSpec('H'),
                  'set' : HeaderElemSpec('H'),
                  'segment' : HeaderElemSpec('H'),
                  'ida' : HeaderElemSpec('H'),
                  'idb' : HeaderElemSpec('H'),
                  'idc' : HeaderElemSpec('H'),
                  'idd' : HeaderElemSpec('H'),
                  'ide' : HeaderElemSpec('H'),
                  'pre' : HeaderElemSpec('H'),
                  'post' : HeaderElemSpec('H'),
                  'kspace_center_column' : HeaderElemSpec('H'),
                  'coil_select' : HeaderElemSpec('H'),
                  'readout_off_center' : HeaderElemSpec('f'),
                  'time_since_last_rf' : HeaderElemSpec('I'),
                  'kspace_center_line' : HeaderElemSpec('H'),
                  'kspace_center_partition' : HeaderElemSpec('H'),
                  'free_parameters' : HeaderElemSpec('H', 4),
                  'sagittal_pos' : HeaderElemSpec('f'),
                  'coronal_pos' : HeaderElemSpec('f'),
                  'transverse_pos' : HeaderElemSpec('f'),
                  'orient_quat' : HeaderElemSpec('f', 4),
                  'channel_id' : HeaderElemSpec('H'),
                  'table_pos_negative' : HeaderElemSpec('H'),
                  'system_type' : HeaderElemSpec('H'),
                  'table_pos_delay' : HeaderElemSpec('H'),
                  'table_pos_x' : HeaderElemSpec('i'),
                  'table_pos_y' : HeaderElemSpec('i'),
                  'table_pos_z' : HeaderElemSpec('i'),
                  'unused1' : HeaderElemSpec('I'),
                  'application_counter' : HeaderElemSpec('H'),
                  'application_mask' : HeaderElemSpec('H'),
                  'checksum' : HeaderElemSpec('I'),
                  'channel_hdr_type_and_len' : HeaderElemSpec('I'),
                  'channel_hdr_meas_uid' : HeaderElemSpec('i'),
                  'channel_hdr_scan_count' : HeaderElemSpec('I'),
                  'channel_hdr_unused2' : HeaderElemSpec('I'),
                  'channel_hdr_sequence_time' : HeaderElemSpec('I'),
                  'channel_hdr_unused3' : HeaderElemSpec('I'),
                  'channel_hdr_channel_id' : HeaderElemSpec('H'),
                  'channel_hdr_unused4' : HeaderElemSpec('H'),
                  'channel_hdr_checksum' : HeaderElemSpec('I'),
                 }
'''Specs for MDH header elements common to all versions'''


mdh_elem_specs_v1 = {'ice_parameters' : HeaderElemSpec('H', 4),
                    }
'''Specs for MDH header elements unique to version 1'''


mdh_elem_specs_v2 = {'ice_parameters' : HeaderElemSpec('H', 24),
                    }
'''Specs for MDH header elements unique to version 2'''


ReadoutHeaderV1 = namedtuple('ReadoutHeaderV1',
                             '''\
                             dma_info
                             meas_uid
                             scan_count
                             timestamp
                             pmu_timestamp
                             eval_info_mask
                             samples_in_scan
                             used_channels
                             line
                             acquisition
                             slice
                             partition
                             echo
                             phase
                             repetition
                             set
                             segment
                             ida
                             idb
                             idc
                             idd
                             ide
                             pre
                             post
                             kspace_center_column
                             coil_select
                             readout_off_center
                             time_since_last_rf
                             kspace_center_line
                             kspace_center_partition
                             ice_parameters
                             free_parameters
                             sagittal_pos
                             coronal_pos
                             transverse_pos
                             orient_quat
                             channel_id
                             table_pos_negative
                             ''')
'''Named tuple for "Version 1" (pre-VD) of the per-readout mini header'''


ReadoutHeaderV2 = namedtuple('ReadoutHeaderV2',
                             '''\
                             dma_info
                             meas_uid
                             scan_count
                             timestamp
                             pmu_timestamp
                             system_type
                             table_pos_delay
                             table_pos_x
                             table_pos_y
                             table_pos_z
                             unused1
                             eval_info_mask
                             samples_in_scan
                             used_channels
                             line
                             acquisition
                             slice
                             partition
                             echo
                             phase
                             repetition
                             set
                             segment
                             ida
                             idb
                             idc
                             idd
                             ide
                             pre
                             post
                             kspace_center_column
                             coil_select
                             readout_off_center
                             time_since_last_rf
                             kspace_center_line
                             kspace_center_partition
                             sagittal_pos
                             coronal_pos
                             transverse_pos
                             orient_quat
                             ice_parameters
                             free_parameters
                             application_counter
                             application_mask
                             checksum
                             ''')
'''Named tuple for "Version 2" (post-VD) of the per-readout mini header'''


chan_elem_specs = {'type_and_len' : HeaderElemSpec('I'),
                   'meas_uid' : HeaderElemSpec('i'),
                   'scan_count' : HeaderElemSpec('I'),
                   'unused2' : HeaderElemSpec('I'),
                   'sequence_time' : HeaderElemSpec('I'),
                   'unused3' : HeaderElemSpec('I'),
                   'channel_id' : HeaderElemSpec('H'),
                   'unused4' : HeaderElemSpec('H'),
                   'checksum' : HeaderElemSpec('I'),
                  }
'''Specs for channel header elements'''


ChannelHeader = namedtuple('ChannelHeader',
                            '''
                            type_and_len
                            meas_uid
                            scan_count
                            unused2
                            sequence_time
                            unused3
                            channel_id
                            unused4
                            checksum
                            ''')
'''Named tuple for channel subheaders included in V2 files'''


def get_struct_fmt(hdr_class, elem_specs):
    '''Get the format string to pass to struct.unpack/pack for this header'''
    struct_fmt = ['<']
    for field in hdr_class._fields:
        spec = elem_specs[field]
        struct_fmt.append(spec.get_struct_fmt_str())
    return ''.join(struct_fmt)


def make_hdr(hdr_class, elem_specs, raw_elems):
    '''Convert result from struct.unpack into the hdr_class'''
    args = []
    struct_idx = 0
    for field in hdr_class._fields:
        spec = elem_specs[field]
        n_elems = spec.n_elems
        if n_elems == 1:
            args.append(raw_elems[struct_idx])
        else:
            args.append(raw_elems[struct_idx:struct_idx+n_elems])
        struct_idx += n_elems
    return hdr_class(*args)


EVAL_INFO_FLAGS = ['ACQEND',
                   'RTFEEDBACK',
                   'HPFEEDBACK',
                   'ONLINE',
                   'OFFLINE',
                   'SYNCDATA',
                   'UNKNOWN6',
                   'UNKNOWN7',
                   'LASTSCANINCONCAT',
                   'UNKNOWN9',
                   'RAWDATACORRECTION',
                   'LASTSCANINMEAS',
                   'SCANSCALEFACTOR',
                   '2NDHADAMARPULSE',
                   'REFPHASESTABSCAN',
                   'PHASESTABSCAN',
                   'D3FFT',
                   'SIGNREV',
                   'PHASEFFT',
                   'SWAPPED',
                   'POSTSHAREDLINE',
                   'PHASCOR',
                   'PATREFSCAN',
                   'PATREFANDIMASCAN',
                   'REFLECT',
                   'NOISEADJSCAN',
                   'SHARENOW',
                   'LASTMEASUREDLINE',
                   'FIRSTSCANINSLICE',
                   'LASTSCANINSLICE',
                   'TREFFECTIVEBEGIN',
                   'TREFFECTIVEEND',
                   'MDS_REF_POSITION',
                   'SLC_AVERAGED'
                   'TAGFLAG1',
                   'CT_NORMALIZE',
                   'SCAN_FIRST',
                   'SCAN_LAST',
                   'UNKNOWN38',
                   'UNKNOWN39',
                   'FIRST_SCAN_IN_BLADE',
                   'LAST_SCAN_IN_BLADE',
                   'LAST_BLADE_IN_TR',
                   'UNKNOWN43',
                   'PACE',
                   'RETRO_LASTPHASE',
                   'RETRO_ENDOFMEAS',
                   'RETRO_REPEATTHISHEARTBEAT',
                   'RETRO_REPEATPREVHEARTBEAT',
                   'RETRO_ABORTSCANNOW',
                   'RETRO_LASTHEARTBEAT',
                   'RETRO_DUMMYSCAN',
                   'RETRO_ARRDETDISABLED',
                   'B1_CONTROLLOOP',
                   'SKIP_ONLINE_PHASCOR',
                   'SKIP_REGRIDDING',
                  ]
'''Bit flags from 'eval_info_mask' in readout header'''


SUPLEMENT_DATA_FLAGS = ['ACQEND',
                        'RTFEEDBACK',
                        'HPFEEDBACK',
                        'SYNCDATA',
                        'REFPHASESTABSCAN',
                        'PHASESTABSCAN',
                        'PHASCOR',
                        'NOISEADJSCAN',
                       ]
'''Flags from eval_info_mask that indicate this is reference/calibration data
'''


SUPLEMENT_DATA_MASK = \
    reduce(lambda mask, flag: mask | (1 << EVAL_INFO_FLAGS.index(flag)) ,
           SUPLEMENT_DATA_FLAGS,
           0)
'''Mask for eval_info_mask flags that indicate this is ref/calibration data'''


class ReadoutV1(object):
    '''Single RF readout from an MR measurement (pre-VD software versions)'''

    _HDR_CLASS = ReadoutHeaderV1

    _HDR_SPECS = deepcopy(mdh_elem_specs)

    _HDR_SPECS.update(mdh_elem_specs_v1)

    _MDH_DMA_LENGTH_MASK = 0x0FFFFFF

    _MDH_PACK_BIT_MASK = 0x02000000

    _MDH_ENABLE_FLAGS_MASK = 0xFC000000

    HDR_STRUCT_FMT = get_struct_fmt(_HDR_CLASS, _HDR_SPECS)

    HDR_STRUCT_SIZE = struct.calcsize(HDR_STRUCT_FMT)

    def __init__(self, hdr, data):
        self.hdr = hdr
        self.data = data
        self._flags = None

    def __str__(self):
        return ''.join(("Header:\n\t",
                        '\n\t'.join(['%s: %s' % (n, v)
                                     for n, v in self.hdr._asdict().items()]),
                        "\nData array shape: %s" % str(self.data.shape),
                        "\nData array type: %s" % str(self.data.dtype),
                      ))

    @property
    def dma_length(self):
        first16 = self.hdr.dma_info & 0xFFFF
        next8 = (self.hdr.dma_info & 0xFF0000) >> 16
        return first16 + (next8 * 2**16)

    def eval_info_is_set(self, flag_name):
        '''Check if the given flag is set in the `eval_info_mask`'''
        idx = EVAL_INFO_FLAGS.index(flag_name)
        return bool(self.hdr.eval_info_mask & (1 << idx))

    @property
    def is_real_acquisition(self):
        """Check if this contains "real" data vs callibration/reference data
        """
        # TODO: This was developed while our eval_info mask handling had some
        #       bugs, and thus needs to be reevaluated
        if self.eval_info_is_set('PATREFANDIMASCAN') or self.eval_info_is_set('PATREFSCAN'):
            return True
        else:
            return (self.hdr.eval_info_mask & SUPLEMENT_DATA_MASK) == 0

    @property
    def is_last_acquisition(self):
        """Returns True if this is the last acquisition
        """
        return self.eval_info_is_set('ACQEND')

    @property
    def eval_info_flags(self):
        '''Return human readable list of flags set in eval_info_mask'''
        if self._flags is not None:
            return self._flags
        self._flags = [x for x in EVAL_INFO_FLAGS if self.eval_info_is_set(x)]
        return self._flags

    @classmethod
    def read_mdh_hdr(klass, src_file):
        hdr_elems = struct.unpack(klass.HDR_STRUCT_FMT,
                                  src_file.read(klass.HDR_STRUCT_SIZE))
        return make_hdr(klass._HDR_CLASS, klass._HDR_SPECS, hdr_elems)

    @classmethod
    def from_file(klass, src_file, no_data=False):
        '''Create a readout by reading from the given source file'''
        hdr = klass.read_mdh_hdr(src_file)

        data_count = 2 * hdr.samples_in_scan
        data_size = 4 * data_count
        dma_size = hdr.dma_info & 0xFFFFFF
        ro_size = klass.HDR_STRUCT_SIZE + data_size
        n_ro = dma_size // ro_size

        if no_data:
            data = None
            src_file.seek(data_size, 1)
        else:
            data = np.fromfile(src_file,
                               dtype=np.float32,
                               count=data_count).view(np.complex64)

        result = [klass(hdr, data)]

        if n_ro < 2 or dma_size % ro_size != 0:
            return result

        for ich in range(1, n_ro):
            cur_hdr = klass.read_mdh_hdr(src_file)
            if no_data:
                cur_data = None
                src_file.seek(data_size, 1)
            else:
                cur_data = np.fromfile(src_file,
                                       dtype=np.float32,
                                       count=data_count).view(np.complex64)
            result.append(klass(cur_hdr, cur_data))

        return result


class ReadoutV2(ReadoutV1):
    '''Single RF readout from an MR measurement (post-VD software versions)'''

    _HDR_CLASS = ReadoutHeaderV2

    _HDR_SPECS = deepcopy(mdh_elem_specs)

    _HDR_SPECS.update(mdh_elem_specs_v2)

    _SUBHDR_TYPE_MASK = 0x000000FF

    _SUBHDR_LEN_RSHIFT = 8

    HDR_STRUCT_FMT = get_struct_fmt(_HDR_CLASS, _HDR_SPECS)

    HDR_STRUCT_SIZE = struct.calcsize(HDR_STRUCT_FMT)

    CHAN_HDR_STRUCT_FMT = get_struct_fmt(ChannelHeader, chan_elem_specs)

    CHAN_HDR_STRUCT_SIZE = struct.calcsize(CHAN_HDR_STRUCT_FMT)

    def __init__(self, hdr, chan_hdr, data):
        super(ReadoutV2, self).__init__(hdr, data)
        self.chan_hdr = chan_hdr

    @classmethod
    def read_chan_hdr(klass, src_file):
        '''Read a single channel header from the current location in src_file
        '''
        hdr_elems = struct.unpack(klass.CHAN_HDR_STRUCT_FMT,
                                  src_file.read(klass.CHAN_HDR_STRUCT_SIZE))
        return make_hdr(ChannelHeader, chan_elem_specs, hdr_elems)

    @classmethod
    def from_file(klass, src_file, no_data=False, chan_idx=None):
        '''Create one or more readouts by reading from the src_file.

        Each channel subheader plus data is treated as its own readout. If
        `chan_idx` is specified than just that channel is returned.
        '''
        hdr = klass.read_mdh_hdr(src_file)
        if hdr.used_channels == 0:
            return [klass(hdr, None, None)]
        result = []
        if chan_idx is not None:
            chans = [chan_idx]
            chan_size = klass.CHAN_HDR_STRUCT_SIZE + (hdr.samples_in_scan * 8)
            src_file.seek(chan_idx * chan_size)
        else:
            chans = range(hdr.used_channels)
        for chan_idx in chans:
            chan_hdr = klass.read_chan_hdr(src_file)
            if no_data:
                data = None
                src_file.seek(hdr.samples_in_scan * 8, 1)
            else:
                data = np.fromfile(src_file,
                                   dtype=np.float32,
                                   count=2 * hdr.samples_in_scan).view(np.complex64)
            result.append(klass(hdr, chan_hdr, data))
        return result


default_counter_order = ('ide',
                         'idd',
                         'idc',
                         'idb',
                         'ida',
                         'acquisition',
                         'echo',
                         'repetition',
                         'set',
                         'segment',
                         'partition',
                         'channel_id',
                         'slice',
                         'phase',
                         'line',
                        )
'''Default slowest-to-fastest order for counters in the k-space array'''


ordinal_counters = ('channel_id',
                   )
'''Counters where we use the order rather than the actual value as an index
'''


class KSpaceSpec(object):
    '''Maps a K-space array to a sequential series of readouts

    Parameters
    ----------

    dim_info : list of tuples
        Each element is a tuple giving the name and size of each dimension
        The last dimension should always be the "readout" dimension.

    idx_map : dict
        Map n-D K-space indices (minus the last index) to 1-D readout indices

    ro_padding : tuple
        Tuple giving the actual readout length and start index
        Can be omitted if there is no zero padding along the readout dim
    '''
    def __init__(self, dim_info, idx_map, ro_padding=None):
        self.dim_info = dim_info
        self.idx_map = idx_map
        self.ro_padding = ro_padding
        self._dim_names = set([d[0] for d in dim_info])

    def pad_dim(self, dim_name, padded_size):
        new_dim_info = []
        # TODO: Warn if padded size doesn't make sense
        for name, size in self.dim_info:
            if name == dim_name:
                new_dim_info.append((name, padded_size))
            else:
                new_dim_info.append((name, size))
        self.dim_info = new_dim_info

    def get_chunk_info(self, fixed=None, bounds=None):
        '''Get information about a subset of k-space

        Parameters
        ----------
        fixed : dict
            Map dimension names to single fixed indices

        bounds : dict
            Map dimension names to tuples of lower/upper bounds

        Returns
        -------
        chunk_shape : tuple
            The shape of the k-space chunk

        ro_map : dict
            Maps 1-D indices of all needed readouts to k-space chunk indices
        '''
        if fixed is None:
            fixed = {}
        else:
            for dim_name in fixed:
                if dim_name not in self._dim_names:
                    raise ValueError("Unknown dimension: %s" % dim_name)
        if bounds is None:
            bounds = {}
        else:
            for dim_name in bounds:
                if dim_name not in self._dim_names:
                    raise ValueError("Unknown dimension: %s" % dim_name)

        # Figure out the chunk of the array we are considering
        lb = []
        ub = []
        out_shape = []
        for dim_name, dim_size in self.dim_info[:-1]:
            if dim_name in fixed:
                fixed_val = fixed[dim_name]
                if not 0 <= fixed_val < dim_size:
                    raise IndexError("Fixed dim '%s' out of bounds" % dim_name)
                lb.append(fixed_val)
                ub.append(fixed_val + 1)
            elif dim_name in bounds:
                lower, upper = bounds[dim_name]
                lower = 0 if lower is None else lower
                upper = dim_size if upper is None else upper
                if not 0 <= lower < upper <= dim_size:
                    raise IndexError("Invalid bounds for dim '%s'" % dim_name)
                lb.append(lower)
                ub.append(upper)
                out_shape.append(upper - lower)
            else:
                lb.append(0)
                ub.append(dim_size)
                out_shape.append(dim_size)
        out_shape.append(self.dim_info[-1][1])

        # Build the map for any data that was actually acquired
        ro_map = {}
        for full_arr_idx in iproduct(*[range(l ,u) for l, u in zip(lb, ub)]):
            map_idx = []
            chunk_idx = []
            for dim_idx, (name, _) in enumerate(self.dim_info[:-1]):
                aidx = full_arr_idx[dim_idx]
                map_idx.append((name, aidx))
                if name not in fixed:
                    chunk_idx.append(aidx - lb[dim_idx])
            map_idx = tuple(sorted(map_idx))
            ro_idx = self.idx_map.get(pickle.dumps(map_idx, pickle.HIGHEST_PROTOCOL))
            if ro_idx is not None:
                ro_map[ro_idx] = tuple(chunk_idx)
        return (out_shape, ro_map)


class KSpaceSizeError(Exception):
    '''Thrown if the computed k-space size is too small for the data'''


class Meas(object):
    '''A single measurement containing some meta data and one or more readouts
    '''

    def __init__(self, src_file, offset, length, meas_id=None, file_id=None,
                 protocol=None, patient=None, version=None, cntr_order=None):
        self._src_file = src_file
        self._offset = offset
        self._length = length
        self._meas_id = meas_id
        self._file_id = file_id
        self._protocol = protocol
        self._patient = patient
        self._version = version
        self._cntr_order = cntr_order
        self._meta = None
        self._k_space_spec = None
        self._mdh_locs = []
        self._mdh_locs_complete = False
        self._ro_per_mdh = []

        # Use default ordering for counters if none was given
        if self._cntr_order is None:
            self._cntr_order = default_counter_order

        # Lineup with our file offset if needed
        if self._src_file.tell() != self._offset:
            self._src_file.seek(self._offset)

        # Read binary descriptors at front of header
        (header_size, n_evps) = struct.unpack('<2I', src_file.read(8))
        self._header_size = header_size
        self._n_evps = n_evps

        # Determine version automatically
        # TODO: Update once proper meta data parsing is included
        if self._version is None:
            vrs_regex = re.compile(r'syngo MR (?P<syngo_version>[A-Z][0-9]+)')
            for name, evp_data in self.meta:
                match = vrs_regex.search(evp_data)
                if match:
                    syngo_version = match.group('syngo_version')
                    if LooseVersion(syngo_version) < LooseVersion('D11'):
                        self._version = 1
                    else:
                        self._version = 2
                    break
            else:
                raise ValueError("Could not automatically determine version")

        if self._version == 1:
            self._readout_class = ReadoutV1
        elif self._version == 2:
            self._readout_class = ReadoutV2
        else:
            raise ValueError("Unknown version: %s" % self._version)

    @property
    def meta(self):
        '''The meta data associated with this measurement'''
        if self._meta is not None:
            return self._meta

        evp_offset = self._offset + 8
        if self._src_file.tell() != evp_offset:
            self._src_file.seek(evp_offset)

        evps = []
        for evp_idx in range(self._n_evps):
            name = _read_cstr(self._src_file)
            (evp_size,) = struct.unpack('<I', self._src_file.read(4))
            evp_data = self._src_file.read(evp_size)
            evps.append((name, evp_data))

        # TODO: handle meta data parsing
        self._meta = evps
        return self._meta

    def gen_readouts(self, no_data=False):
        '''Generate readouts one at a time as stored in the file'''
        mdh_offset = self._offset + self._header_size
        mdh_idx = 0

        while True:
            if self._src_file.tell() != mdh_offset:
                self._src_file.seek(mdh_offset)
            readouts = self._readout_class.from_file(self._src_file, no_data)
            if readouts[0].is_last_acquisition:
                self._mdh_locs_complete = True
                break
            mdh_idx += 1
            if len(self._mdh_locs) < mdh_idx:
                self._mdh_locs.append(mdh_offset)
                self._ro_per_mdh.append(len(readouts))
            mdh_offset += readouts[0].dma_length
            for readout in readouts:
                yield readout

    def get_k_space_spec(self):
        '''Get mapping from "counters" to k-space axes and their size

        This requires a full pass through the file so we can find all varying
        counters, and thus it can be slow on large files.
        '''
        if self._k_space_spec != None:
            return self._k_space_spec

        # Iterate through all measurement headers and keep track of how the
        # various "counters" vary or duplicate one another
        non_dupes = {}
        indices = deque()
        counter_sets = [set() for name in self._cntr_order]
        found_counters = None
        cntr_pack_fmt = None
        ro_gen = self.gen_readouts(no_data=True)
        first_ro = None
        ro_idx = -1
        real_acq_count = 1
        while first_ro is None or not first_ro.is_real_acquisition:
            ro_idx += 1
            first_ro = next(ro_gen)
        samples_in_scan = first_ro.hdr.samples_in_scan
        for ro in ro_gen:
            ro_idx += 1
            if ro.is_last_acquisition:
                break
            if not ro.is_real_acquisition:
                continue
            real_acq_count += 1
            if samples_in_scan != ro.hdr.samples_in_scan:
                raise ValueError("The samples_in_scan changed: %d vs %d"
                                 % (samples_in_scan,
                                    ro.hdr.samples_in_scan))
            cntr_vals = []
            for cidx, cntr in enumerate(self._cntr_order):
                if hasattr(ro.hdr, cntr):
                    cntr_val = getattr(ro.hdr, cntr)
                elif hasattr(ro, 'chan_hdr') and hasattr(ro.chan_hdr, cntr):
                    cntr_val = getattr(ro.chan_hdr, cntr)
                else:
                    continue # TODO: Exception or warning here?
                counter_sets[cidx].add(cntr_val)
                cntr_vals.append((cntr, cntr_val))
            n_cntrs = len(cntr_vals)
            if found_counters is None:
                found_counters = [x[0] for x in cntr_vals]
                cntr_pack_fmt = '%dH' % n_cntrs
            packed_cntrs = struct.pack(cntr_pack_fmt,
                                       *(x[1] for x in cntr_vals))
            indices.append((packed_cntrs, ro_idx))
            # TODO: More efficient way to handle duplicate counters?
            for idx, (name, val) in enumerate(cntr_vals):
                if not name in non_dupes:
                    non_dupes[name] = set()
                for name2, val2 in cntr_vals:
                    if val != val2 or name == name2:
                        non_dupes[name].add(name2)
        mid_dt = datetime.now()
        # Pull out info about the varying counters
        varying_set = set()
        varying_counters = []
        varying_values = {}
        ordinal_lists = {}
        for idx, counter_name in enumerate(self._cntr_order):
            if counter_name in ordinal_counters:
                count = len(counter_sets[idx])
            else:
                count = max(counter_sets[idx]) + 1
            if count > 1:
                if counter_name in ordinal_counters:
                    ordinal_lists[counter_name] = sorted(list(counter_sets[idx]))
                varying_set.add(counter_name)
                varying_counters.append((counter_name, count))
                varying_values[counter_name] = sorted(counter_sets[idx])

        # Remove any varying counters that are duplicates
        all_dupes = set()
        for name, _ in reversed(varying_counters):
            if name in all_dupes:
                continue
            dupes = varying_set.difference(non_dupes[name])
            all_dupes.update(dupes)
            if len(dupes) != 0:
                varying_counters = [item for item in varying_counters
                                    if not item[0] in dupes]

        # Basic shape_info and sanity check
        full_count = 1
        shape = []
        for dim_name, dim_size in varying_counters:
            full_count *= dim_size
            shape.append(dim_size)
        if full_count < real_acq_count:
            raise KSpaceSizeError("K-space shape %s is too small "
                                  "(%d readouts > %d possible indices" %
                                  (shape, real_acq_count, full_count))

        # Create map from our multi-dimensional k-space indices to the
        # sequential readout indices
        idx_map = {}
        while len(indices) >= 1:
            k_spc_idx = []
            packed_cntrs, ro_idx = indices.popleft()
            cntr_vals = struct.unpack(cntr_pack_fmt, packed_cntrs)
            for name, _ in varying_counters:
                cntr_idx = found_counters.index(name)
                if name in ordinal_counters:
                    idx_val = ordinal_lists[name].index(cntr_vals[cntr_idx])
                else:
                    idx_val = cntr_vals[cntr_idx]
                k_spc_idx.append((name, idx_val))
            k_spc_idx = tuple(sorted(k_spc_idx))
            idx_map[pickle.dumps(k_spc_idx, pickle.HIGHEST_PROTOCOL)] = ro_idx

        end_dt = datetime.now()

        # Build and return the KSpaceSpec object
        dim_info = varying_counters + [('readout', samples_in_scan)]
        self._k_space_spec = KSpaceSpec(dim_info, idx_map)
        return self._k_space_spec

    def _insert_ro(self, readout, arr, arr_idx):
        expected_ro = arr.shape[-1]
        rdata = readout.data
        actual_ro = len(rdata)
        if actual_ro < expected_ro:
            # Async echo
            if readout.eval_info_is_set('REFLECT'):
                arr[arr_idx][:actual_ro] = rdata[::-1]
            else:
                arr[arr_idx][expected_ro - actual_ro:] = rdata
        else:
            if readout.eval_info_is_set('REFLECT'):
                arr[arr_idx] = rdata[::-1]
            else:
                arr[arr_idx] = rdata

    def _fill_with_seek(self, spec, ro_map, arr):
        max_ro_per_mdh = np.cumsum(self._ro_per_mdh) - 1
        ro_indices = sorted(ro_map.keys())
        ro_buf = None
        for ro_idx in ro_indices:
            if ro_buf is None or ro_idx > last_idx:
                mdh_idx = np.searchsorted(max_ro_per_mdh, ro_idx)
                offset = self._mdh_locs[mdh_idx]
                if self._src_file.tell() != offset:
                    self._src_file.seek(offset)
                ro_buf = self._readout_class.from_file(self._src_file)
                if mdh_idx == 0:
                    first_idx = 0
                else:
                    first_idx = max_ro_per_mdh[mdh_idx - 1] + 1
                last_idx = max_ro_per_mdh[mdh_idx]
                curr_idx = first_idx
            buf_idx = ro_idx - first_idx
            arr_idx = ro_map[ro_idx]
            ro = ro_buf[buf_idx]
            self._insert_ro(ro_buf[buf_idx], arr, arr_idx)

    def _fill_seq(self, spec, ro_map, arr):
        ro_indices = deque(sorted(ro_map.keys()))
        curr_ro_idx = ro_indices.popleft()
        for ro_idx, readout in enumerate(self.gen_readouts()):
            if ro_idx == curr_ro_idx:
                arr_idx = ro_map[ro_idx]
                self._insert_ro(readout, arr, arr_idx)
                if len(ro_indices) == 0:
                    break
                curr_ro_idx = ro_indices.popleft()

    def get_k_space(self, spec=None, fixed=None, bounds=None):
        '''Get (some of) the k-space array

        You can fix some of the counters to only get a subset on the k-space
        array, and thus constrain memory use.
        '''
        if spec is None:
            spec = self.get_k_space_spec()

        # Figure out the indices of the readouts we need
        out_shape, ro_map = spec.get_chunk_info(fixed, bounds)

        # Create a zeroed k-space array and fill in the acquired data
        k_spc = np.zeros(out_shape, dtype=np.complex64)
        if self._mdh_locs_complete:
            self._fill_with_seek(spec, ro_map, k_spc)
        else:
            self._fill_seq(spec, ro_map, k_spc)

        return k_spc


class MeasFile(object):
    '''A single Twix file, which may include one or more measurements.

    Usually the last measurement is the actual experiment while any earlier
    measurements are some sort of preparation/calibration.
    '''
    def __init__(self, src, version=None):
        if isinstance(src, str):
            self._src_file = open(src, 'rb')
        else:
            self._src_file = src
        self._meas = []
        (test,) = struct.unpack('<I', self._src_file.read(4))
        if test != 0:
            self._meas.append(Meas(self._src_file,
                                   0,
                                   os.fstat(self._src_file.fileno()).st_size,
                                   version=version)
                             )
        else:
            if version is not None:
                assert version == 2
            version = 2
            (n_meas,) = struct.unpack('<I', self._src_file.read(4))
            for meas_idx in range(n_meas):
                (meas_id,
                 file_id,
                 offset,
                 length,
                 patient,
                 protocol) = struct.unpack('<2I2Q64s64s',
                                           self._src_file.read(152))
                patient = patient.rstrip(b'\0')
                protocol = protocol.rstrip(b'\0')
                self._meas.append(Meas(self._src_file,
                                       offset,
                                       length,
                                       meas_id,
                                       file_id,
                                       protocol,
                                       patient,
                                       version=version)
                                 )

    @property
    def n_meas(self):
        '''The number of measurements in this file'''
        return len(self._meas)

    def get_meta(self, meas_idx=-1):
        '''Get meta data from the measurement at `meas_idx`'''
        return self._meas[meas_idx].meta

    def gen_readouts(self, meas_idx=-1):
        '''Generate indivdual readouts from the measurement at `meas_idx`'''
        for ro in self._meas[meas_idx].gen_readouts():
            yield ro

    def get_k_space_spec(self, meas_idx=-1):
        '''Get info about the shape of the full k-space array'''
        return self._meas[meas_idx].get_k_space_spec()

    def get_k_space(self, meas_idx=-1, spec=None, fixed=None, bounds=None):
        return self._meas[meas_idx].get_k_space(spec, fixed, bounds)
