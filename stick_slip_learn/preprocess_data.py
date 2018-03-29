
##-----------------
##-----------------
# PREPROCESS DATA: RESCALING THE DATA USING STANDARD SCALER FROM SCIKIT LEARN
#
##

def rescale_data(data, **kwargs):

    import numpy as np
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)

    # scaler_mean = scaler.mean_
    # scaler_std = scaler.std_

    data = np.array(data)

    if 'verbose' in kwargs:
    # optional verbose output
        if kwargs['verbose'] == True:
            print("\n","".join(["*"]*20), "printing scaler.fit(data)","\n")
            print(scaler)
            print("\n","".join(["*"]*20), "printing scaler.mean_","\n")
            print(scaler.mean_)
            print("\n","".join(["*"]*20), "printing scaler.std_","\n")
            print(scaler.std_)
            print("\n","".join(["*"]*20), "printing scaler.tranform(all_records)","\n")
            print(data_standardized)

    return data_standardized, scaler

##-----------------
##-----------------
# PREPROCESS DATA: FORMAT THE DATA INTO A LEARNING FORMAT USED FOR TRAINING
#
##

def learning_format(force_contacts, particle_contacts, particle_positions, n_dim, n_feat_part, n_feat_cont_max, pos_file_rec_len, **kwargs):

    import numpy as np
    learning_data = []

    if len(force_contacts) != len(particle_contacts) or len(force_contacts) != len(particle_positions):
        raise ValueError('Dimension mismatch.')

    if "verbose" in kwargs:
        verbose = kwargs["verbose"]
    else:
        verbose = False

    ##-----------------
    # For each snapshot of forces, positions and contacts glue these data tohgether
    # to form a learning format used later for training or testing or prediction
    ##-----------------
    for idx in range(len(force_contacts)):

        if verbose: print("processing zone ", idx)

        force_on_particle_data = []
        n_features = n_feat_part*n_feat_cont_max

        ##-----------------
        # data has to have particular dimensions for learning
        ##-----------------
        fc_shape = [n_dim,n_feat_part,n_feat_cont_max]
        pc_shape = [n_feat_part,n_feat_cont_max]
        pp_shape = [pos_file_rec_len,n_feat_part]

        force_contacts_snapshot = reshape_data(force_contacts[idx], fc_shape)
        particle_contacts_snapshot = reshape_data(particle_contacts[idx], pc_shape)
        particle_positions_snapshop = reshape_data(particle_positions[idx], pp_shape, transpose = True)

        ##-----------------
        # Glue data together.
        ##-----------------
        for particle in range(n_feat_part):
            for contact in range(n_feat_cont_max):
                particle2 = particle_contacts_snapshot[particle, contact] - 1
                if particle2 != -1:
                    force = force_contacts_snapshot[:, particle, contact]
                    position_p1 = particle_positions_snapshop[particle][:n_dim]
                    position_p2 = particle_positions_snapshop[particle2][:n_dim]

                    force_on_particle_data.append(np.append(position_p1,force))
                    force_on_particle_data.append(np.append(position_p2,[-x for x in force]))

        ##-----------------
        # The maximum size of the data is n_features = n_feat_part*n_feat_cont_max
        # where n_feat_part is the number of particles and n_feat_cont_max is
        # the maximum possible number of contacts.
        # To guarantee the input data length, assume 0.0 entries if
        # n_features>len(force_on_particle_data)
        ##-----------------

        for additional in range(n_features-len(force_on_particle_data)):
            force_on_particle_data.append(np.array([0.0 for i in range(n_dim*2)]))

        ##-----------------
        # rescale data so they are centered and have a variance == 1.0 for
        # faster training.
        ##-----------------
        force_on_particle_data, scaler = rescale_data(force_on_particle_data)

        learning_data.append(force_on_particle_data)

    return learning_data

##-----------------
##-----------------
# RESHAPING THE DATA TO HAVE A shape SHAPE
#
##
def reshape_data(data_to_reshape, shape, **kwarg):
    import numpy as np

    data_to_reshape = np.array(data_to_reshape)
    data_to_reshape = data_to_reshape.reshape(shape)

    if "transpose" in kwarg:
        if kwarg["transpose"]:
            data_to_reshape = data_to_reshape.T

    return data_to_reshape

##-----------------
##-----------------
# SELECTING THE DATA WITH SPECIFIC INDICATORS
#
##

def select_data_by_ind(data, indicators, which_ind):

    data_to_return = []

    if len(indicators) != len(data):
        raise ValueError("Dimension mismatch, len(indicators): "+str(len(indicators))+" len(data): "+str(len(data)))

    for idx in range(len(indicators)):
        if indicators[idx] in which_ind:
            data_to_return.append(data[idx])

    return data_to_return
