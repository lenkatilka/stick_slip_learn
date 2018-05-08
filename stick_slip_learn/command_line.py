

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
                'batch_size': 2000,
                'which_ind': {0,1,2}}

    filename_force = path + "force_contact.bin"
    filename_contacts = path + "contact_list.bin"
    filename_particles = path + "part.bin"
    filename_slip = path + "slip_interval_indicator.dat"

    training = False
    prob_slip = True

    if training :
        building_model = stick_slip_learn.classification_model(arguments)
        warm_start = False

        for start_rec in range(arguments['min_record'], arguments['max_record'], arguments['batch_size']):
            start_time = time.time()
            filename_batch = "../stick_slip/training_features_labels_"+str(start_rec)+"_"+str(start_rec+arguments['batch_size']-1)+".h5"
            # building_model.get_all_data(filename_force, filename_contacts, filename_particles, filename_slip)
            building_model.get_train_data(filename_batch, warm_start=warm_start)
            warm_start = True
            end_time = time.time()

            print("".join(["-"]*108)+"\nIt took ", end_time - start_time, " to get all data\n"+"".join(["-"]*108))

            start_time = time.time()
            building_model.train_model()
            end_time = time.time()

            if (start_rec%10000 == 601) and (start_rec != 601):
                stick_slip_learn.save_model(building_model.fitted_model, "../stick_slip/models/rf_"+str(start_rec)+"_"+str(start_rec+arguments['batch_size']-1)+".pickle")


        print("".join(["-"]*108)+"\nIt took ", end_time - start_time, " to train the data\n"+"".join(["-"]*108))

        stick_slip_learn.save_model(building_model.fitted_model,"../stick_slip/models/final_rf_model.pickle")
        return building_model

    elif prob_slip:
        trained_model = stick_slip_learn.load_model("../stick_slip/models/rf_290601_292600.pickle")
        class_data = stick_slip_learn.classification_model(arguments)
        warm_start = False
        predicted_prob = []

        for start_rec in range(arguments['min_record'], 290602, arguments['batch_size']):
            start_time = time.time()
            filename_batch = "../stick_slip/training_features_labels_"+str(start_rec)+"_"+str(start_rec+arguments['batch_size']-1)+".h5"
            class_data.get_all_data(filename_force, filename_contacts, filename_particles, filename_slip)

            predicted_prob = predicted_prob + trained_model.predict_proba(class_data.training_data)

            if not isinstance(predicted_prob, list):
                raise ValueError("predicted probabilities are not in list")

        time = 601
        with open("predicted_prob_of_slip.dat", "w") as fout:
            for prob in predicted_prob:
                fout.write(str(time)+ " " + str(prob)+ "\n")
                time += 1





if __name__=="__main__":
    main()
