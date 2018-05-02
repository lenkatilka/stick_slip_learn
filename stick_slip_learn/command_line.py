

def main():

    import stick_slip_learn
    import numpy as np
    import time
    path = "data/"


    arguments = { 'n_feat_part':2620,
                'n_feat_cont_max': 9,
                'pos_file_rec_len': 8,
                'offset_ind': 601,
                'min_record': 601,
                'max_record': 300000,
                'n_dim': 2,
                'batch_size': 2000}

    filename_force = path + "force_contact.bin"
    filename_contacts = path + "contact_list.bin"
    filename_particles = path + "part.bin"
    filename_slip = path + "slip_interval_indicator.dat"

    building_model = stick_slip_learn.classification_model(arguments)
    warm_start = False

    for start_rec in range(arguments['min_record'], arguments['max_record'], arguments['batch_size']):
        start_time = time.time()
        filename_batch = "../training_features_labels_"+str(start_rec)+"_"+str(start_rec+arguments['batch_size']-1)+".h5"
        # building_model.get_all_data(filename_force, filename_contacts, filename_particles, filename_slip)
        building_model.get_train_data(filename_batch, warm_start=warm_start)
        warm_start = True
        end_time = time.time()

        print("".join(["-"]*108)+"\nIt took ", end_time - start_time, " to get all data\n"+"".join(["-"]*108))

        start_time = time.time()
        building_model.train_model()
        end_time = time.time()



    print("".join(["-"]*108)+"\nIt took ", end_time - start_time, " to train the data\n"+"".join(["-"]*108))

    return building_model

if __name__=="__main__":
    main()
