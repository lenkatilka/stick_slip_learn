##-----------------
##-----------------
# CLASS FOR THE TRAINING OF THE MODEL; TRAINING IS PERFORMED ON THE
# DATA READ IN SEQUENCES -- NOT ALL AT ONCE -- AND WARM START IS USED
##
##
class classification_model():

    import sklearn

    def __init__(self,arguments, tol=0.0001, C=0.01, random_state=37845, max_iter=10000, warm_start=False):
        self.tol = tol
        self.C = C
        self.random_state = random_state
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.n_feat_part = arguments['n_feat_part']
        self.n_feat_cont_max = arguments['n_feat_cont_max']
        self.pos_file_rec_len = arguments['pos_file_rec_len']
        self.offset_ind = arguments['offset_ind']
        self.n_dim = arguments['n_dim']
        self.min_record = arguments['min_record']
        self.max_record = arguments['max_record']
        self.batch_size = arguments['batch_size']
        self.batch_start = arguments['batch_start']

    ##-----------------
    ##-----------------
    # TRAINING THE LOGISTIC REGRESSION MODEL
    ##
    ##
    def train_log_reg(self, **kwargs):
        from sklearn.linear_model import LogisticRegression

        logistic = LogisticRegression(penalty='l2',solver="saga",tol=self.tol, C=self.C, random_state=self.random_state, max_iter=self.max_iter, warm_start=self.warm_start)

        self.fitted_model = logistic.fit(self.training_data, self.training_labels)
        return

    ##-----------------
    ##-----------------
    # TRAINING THE LOGISTIC REGRESSION MODEL
    ##
    ##
    def train_rand_forest(self, **kwargs):
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(criterion = "entropy", warm_start = self.warm_start, **kwargs   )
        self.fitted_model = rf.fit(self.training_data, self.training_labels)

        return

    ##-----------------
    ##-----------------
    # PREDICTION FROM THE TRAINED MODEL
    ##
    ##
    def predict(self, prediction_data):
        return self.fitted_model.predict(prediction_data)

    ##-----------------
    ##-----------------
    # SET UP ALL THE DATA FOR TRAINING
    ##
    ##
    def get_all_data(self, filename_force, filename_contacts, filename_particles, filename_slip):

        while self.batch_start < self.max_record:
            self.get_batch_data(filename_force, filename_contacts, filename_particles, filename_slip)
            self.batch_start += self.batch_size

        return

    ##-----------------
    ##-----------------
    # SAVE THE DATA FOR TRAINING
    ##
    ##
    def save_data(self, filename, method = 'hdf5'):

        ##-----------------
        # Creating the dataset with features and labels
        ##-----------------
        if method == 'hdf5':

            import h5py

            data_h5 = h5py.File(filename+".h5", 'w')
            data_h5.create_dataset('features', data = self.training_data)
            data_h5.create_dataset('labels', data = self.training_labels)

            data_h5.close()

        else:
            raise IOError('[Errno 5] Input/Output error: only hdf5 output format permitted at this time.')

        return

    ##-----------------
    ##-----------------
    # SET UP BATCH DATA FOR TRAINING
    ##
    ##

    def get_batch_data(self, filename_force, filename_contacts, filename_particles, filename_slip, verbose = False):

        import stick_slip_learn

        keywords = {'min_record': self.min_record, 'max_record': self.max_record}
        keywords_cont = {'min_record': self.min_record, 'max_record':self. max_record, 'dtype': 'int32'}
        ##-----------------
        # Creating the force data object
        ##-----------------
        if verbose:
            print("".join(['-']*50),"\n creating the force data object\n", "".join(['-']*50))
        force_object = self.create_data_object(filename = filename_force, rec_len = self.n_feat_cont_max*self.n_dim*self.n_feat_part,**keywords)

        ##-----------------
        # Creating the particle data object
        ##-----------------
        if verbose:
            print("".join(['-']*50),"\n creating the particle data object\n", "".join(['-']*50))
        particle_object = self.create_data_object(filename = filename_particles, rec_len = self.n_feat_part*self.pos_file_rec_len,**keywords)

        ##-----------------
        # Creating the contatcs data object
        ##-----------------
        if verbose:
            print("".join(['-']*50),"\n creating the contacts data object\n", "".join(['-']*50))
        contacts_object = self.create_data_object(filename = filename_contacts, rec_len = self.n_feat_part*self.n_feat_cont_max,**keywords_cont)

        ##-----------------
        # Creating the indicators array to get labels for training
        ##-----------------
        force_object.get_indicators_from_file(filename_slip, offset= self.offset_ind)

        ##-----------------
        # Creating the learning format from force, particle and contact objects
        ##-----------------
        if verbose:
            print("".join(['-']*50),"\n creating the learning format\n", "".join(['-']*50))
        all_data = stick_slip_learn.learning_format(force_object.data, contacts_object.data, particle_object.data, self.n_dim, self.n_feat_part, self.n_feat_cont_max, self.pos_file_rec_len, verbose=True)

        ##-----------------
        # Selection of the data according to the labels we want to train on
        ##-----------------
        self.training_data = stick_slip_learn.select_data_by_ind(all_data, force_object.indicators_for_data, which_ind={0,2})
        self.training_labels = stick_slip_learn.select_data_by_ind(force_object.indicators_for_data, force_object.indicators_for_data, which_ind={0,2})


        filename_batch = "training_features_labels_"+str(self.batch_start)+"_"+str(self.batch_start+self.batch_size-1)
        self.save_data(filename_batch)
        return

    ##-----------------
    ##-----------------
    # TRAIN A SELECTED MODEL; THIS VERSION OF THE CODE USES ONLY LOGISTIC REGRESSION
    ##
    ##

    def train_model(self, which = "random_forest", verbose=True):
        import numpy as np

        if len(set(self.training_labels)) < 2:
            print("Nothing to train on for this data chunk. Reccomended to increase the data batch size.")
            return

        ##-----------------
        # find the dimension of the training_data data
        ##-----------------

        n_records, n_features = np.array(self.training_data).shape[0:2]
        if verbose:
            print("n_records, n_features: ", n_records, n_features)

        ##-----------------
        # reshape the training data and labels for learning
        ##-----------------
        self.training_data = np.array(self.training_data).reshape([n_records,n_features*2*self.n_dim])
        self.training_labels = np.array(self.training_labels).reshape(len(self.training_labels),)

        ##-----------------
        # call a scikit-learn logistic regression library with specific parameters
        ##-----------------
        if verbose:
            print("".join(['-']*50),"\n training "+ which + " \n", "".join(['-']*50))

        if which == "logistic_regression":
            self.train_log_reg()

        if which == "random_forest":
            kwargs = {'n_estimators': 20}
            self.train_rand_forest(**kwargs)

        return

    ##-----------------
    ##-----------------
    # CREATE A DATA OBJECT FROM A FILE
    ##
    ##
    def create_data_object(self, filename, rec_len, **kwargs):
        import stick_slip_learn

        data_object = stick_slip_learn.data_file(filename)
        data_object.import_f_bin(rec_len = rec_len, **kwargs)

        return data_object
