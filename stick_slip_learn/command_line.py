

def main():

    import stick_slip_learn
    import numpy as np
    path = "/Users/lenkatilka/Research/documents/stick_slip/slip_event/"


    n_feat_part = 2620
    n_feat_cont_max = 9
    pos_file_rec_len = 8
    offset_ind = 5000
    min_record = 0
    max_record = 2000
    n_dim = 2


    filename_force = path + "force_contact_5000_8000.bin"
    filename_contacts = path + "contact_list_5000_8000.bin"
    filename_particles = path + "part_5000_8000.bin"
    filename_slip = path + "slip_interval_indicator.dat"

    building_model = stick_slip_learn.classification_model(n_feat_part, n_feat_cont_max, pos_file_rec_len, offset_ind, n_dim, min_record, max_record)
    building_model.get_all_data(filename_force, filename_contacts, filename_particles, filename_slip, verbose = False)
    building_model.train_model()

    return building_model
