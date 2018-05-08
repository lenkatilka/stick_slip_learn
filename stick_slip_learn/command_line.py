

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

    training = True
    prob_slip = False

    if training :
        building_model = stick_slip_learn.classification_model(arguments)
        warm_start = False

        for start_rec in range(arguments['min_record'], arguments['max_record'], arguments['batch_size']):
            start_time = time.time()

            end_rec = min(start_rec+arguments['batch_size']-1, arguments['max_record'])
            filename_batch = "../stick_slip/training_features_labels_"+str(start_rec)+"_"+str(end_rec)+".h5"

            building_model.get_batch_data(filename_force, filename_contacts, filename_particles, filename_slip)
            # building_model.get_train_data(filename_batch, warm_start=warm_start)
            warm_start = True
            end_time = time.time()

            print("".join(["-"]*108)+"\nIt took ", end_time - start_time, " to get all data\n"+"".join(["-"]*108))

            try :
                start_time = time.time()
                building_model.train_model()
                end_time = time.time()

                stick_slip_learn.save_model(building_model.fitted_model, "../stick_slip/models/rf_"+str(start_rec)+"_"+str(end_rec)+".pickle")
            except:
                pass

        print("".join(["-"]*108)+"\nIt took ", end_time - start_time, " to train the data\n"+"".join(["-"]*108))

        stick_slip_learn.save_model(building_model.fitted_model,"../stick_slip/models/final_rf_model.pickle")
        return building_model

    elif prob_slip:
        trained_model = stick_slip_learn.load_model("../stick_slip/models/rf_290601_292600.pickle")
        class_data = stick_slip_learn.classification_model(arguments)
        warm_start = False
        predicted_prob = []

        fout = open("predicted_prob_of_slip.dat", "w")
        time_out = arguments['min_record']

        for start_rec in range(arguments['min_record'], 290602, arguments['batch_size']):
            start_time = time.time()
            class_data.min_record = start_rec
            class_data.get_batch_data(filename_force, filename_contacts, filename_particles, filename_slip)

            n_records, n_features = np.array(class_data.training_data).shape[0:2]
            class_data.training_data = np.array(class_data.training_data).reshape([n_records,n_features*2*class_data.n_dim])
            predicted_prob = trained_model.predict_proba(class_data.training_data)
            predicted_prob = predicted_prob[:,1]

#            if not isinstance(predicted_prob, list):
#                print(predicted_prob)
#                raise ValueError("predicted probabilities are not in list")


            for prob in predicted_prob:
                fout.write(str(time_out)+ " " + str(prob)+ "\n")
                time_out += 1





if __name__=="__main__":
    main()
