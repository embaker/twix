====
Twix
====

This is a python package for reading Siemens "TWIX" (aka. "meas", "mdh") files. This is 
the format used to store raw RF measurement data on Siemens MRI instruments.


Quick Start
===========

The first step is to create a `MeasFile` object:

..
    from twix import MeasFile
    mf = MeasFile('path/to/my/file/meas_MID1111_some_proto_FID22222.dat')

The majority of a twix file is composed of "readout" lines, each with a small set of 
associated meta data. We can iterate through the readout lines one at a time to avoid 
using a large amount of RAM. Here we simply grab the first readout and the break out 
of the for loop. We can get a quick overview of the readout meta data by just 
printing the object.

..
    first_ro = None
    for readout in mf.gen_readouts():
        first_ro = readout
        break
    print first_ro
    Header:
        dma_info: 4320
        meas_uid: 1428
        scan_count: 1
        timestamp: 11392132
        pmu_timestamp: 92128154
        system_type: 0
        table_pos_delay: 26011
        table_pos_x: 0
        table_pos_y: 0
        table_pos_z: -1761228
        unused1: 0
        eval_info_mask: 268435464
        samples_in_scan: 512
        used_channels: 1
        line: 0
        acquisition: 0
        slice: 0
        partition: 0
        echo: 0
        phase: 0
        repetition: 0
        set: 0
        segment: 0
        ida: 0
        idb: 0
        idc: 0
        idd: 0
        ide: 0
        pre: 0
        post: 0
        kspace_center_column: 256
        coil_select: 0
        readout_off_center: 0.0
        time_since_last_rf: 0
        kspace_center_line: 86
        kspace_center_partition: 0
        sagittal_pos: -42.0339012146
        coronal_pos: -16.5153808594
        transverse_pos: -78.2305908203
        orient_quat: (0.7042522430419922, 0.06347236037254333, -0.06347236037254333, 0.7042522430419922)
        ice_parameters: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        free_parameters: (0, 0, 0, 0)
        application_counter: 0
        application_mask: 0
        checksum: 0
    Data array shape: (512,)
    Data array type: complex64


Usually, what you actually want to work with is a multidimensional k-space array rather
than a bunch of individual readout lines. Finding the correct mapping from readout lines
to a k-space array for an arbitrary pulse sequence is not always possible. Regardless, 
it can be helpful to use the `get_k_space_spec` method to see what the best guess is for 
this mapping:

..
    spec = mf.get_k_space_spec()
    print spec.dim_info
    [('slice', 21), ('line', 171), ('readout', 512)]


In this case we have data from a relatively simple pulse sequence, and the returned "spec" 
is actually the correct mapping. We can request the actual k-space array, or a portion of 
it, with the `get_k_space` method.

..
    kspc = mf.get_k_space()
    print kspc.shape
    (21, 171, 512)

If you want just a portion the k-space array (i.e. due to memory constraints) you can fix 
on or more of the "counter" values.

..
    kspc = mf.get_k_space(fixed={'slice': 12})
    print kspc.shape
    (171, 512)

TODO: Add info about manually updating the k-space spec (changing order or adding zero-padding)
