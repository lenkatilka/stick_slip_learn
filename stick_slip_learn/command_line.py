

def main():

    import stick_slip_learn
    import numpy as np
    from pandas import HDFStore, DataFrame
    import time
    path = "/Users/lenkatilka/Research/documents/stick_slip/slip_event/"


    arguments = { 'n_feat_part':2620,
                'n_feat_cont_max': 9,
                'pos_file_rec_len': 8,
                'offset_ind': 8000,
                'min_record': 0,
                'max_record': 4000,
                'n_dim': 2,
                'batch_size': 2000}

    filename_force = path + "force_contact.bin"
    filename_contacts = path + "contact_list.bin"
    filename_particles = path + "part.bin"
    filename_slip = path + "slip_interval_indicator.dat"

    building_model = stick_slip_learn.classification_model(arguments)

    start_time = time.time()
    building_model.get_all_data(filename_force, filename_contacts, filename_particles, filename_slip)
    end_time = time.time()

    print("".join(["-"]*108)+"\nIt took ", end_time - start_time, " to get all data\n"+"".join(["-"]*108))

    start_time = time.time()



    end_time = time.time()
    print("".join(["-"]*108)+"\nIt took ", end_time - start_time, " to store all data\n"+"".join(["-"]*108))

    start_time = time.time()
    building_model.train_model()
    end_time = time.time()

    print("".join(["-"]*108)+"\nIt took ", end_time - start_time, " to train the data\n"+"".join(["-"]*108))

    return building_model
