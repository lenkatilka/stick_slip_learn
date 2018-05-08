##-----------------
##-----------------
# CLASS THAT HOLDS DATA FOR FORCES OR PARTICLE POSITIONS OR PARTICLE CONTACTS
# USED FOR TRAINING
##
##
class data_file():

    def __init__(self, filename):
        self.filename = filename
        self.indicators_for_data = []
        self.data = []
        self.average_over = 0
        self.prefix_interval = 0
        self.wall_length = 0
        self.threshold_size = 0
        self.rec_len = 0

    ##-----------------
    ##-----------------
    # IMPORT DATA FROM BINARY FORTRAN FILE
    ##
    ##
    def import_f_bin(self, rec_len, **kwargs):

        import warnings
        ##-----------------
        ## setting some default values if the following arguments are not specified
        ##-----------------

        if ('min_record' in kwargs):
            self.min_record = kwargs['min_record']
        else:
            self.min_record = 0
            warnings.warn("No beginning of the record specified; reading from record No. 1.")

        if ('max_record' in kwargs):
            self.max_record = kwargs['max_record']

        else:
            self.max_record = 1000
            warnings.warn("No end of the record specified; reading till record No. 1000.")

        if ('dtype' in kwargs):
            dtype = kwargs['dtype']

        else:
            dtype = "float64"
            warnings.warn("No data type specified; assuming float64.")

        ##-----------------
        ##
        ##-----------------


        ##-----------------
        ## Check the input values
        ##-----------------

        if self.max_record <= self.min_record:
            raise ValueError("Beginning of the record cannot be after the end of record.")

        if self.max_record <= 0 or self.min_record <0:
            raise ValueError("Record beginning or record end is set to non-positive value.")

        if rec_len <= 0:
            raise ValueError("Record length cannot be a non-positive number.")
        ##-----------------
        ##
        ##-----------------



        self.rec_len = rec_len
        generate_record = self.record_from_file(rec_len, dtype)

        ##-----------------
        # create data chunk by reading the binary file and appending the
        # records that have an indicator specified in which_ind set
        ##-----------------

        for record_id in range(self.min_record, self.max_record):

            record = next(generate_record)
            # consider only records with offset >= self.min_record
            if record_id < self.min_record:
                continue

            self.data.append(record)
        ##-----------------
        ##
        ##-----------------
        return

    ##-----------------
    ##-----------------
    # GENERATOR FOR READING A BINARY FILE IN A STREAM
    ##
    ##
    def record_from_file(self, rec_len, dtype):
        import numpy as np
        with open(self.filename, 'r') as fin:
            while True:
                data_to_return = np.fromfile(fin, dtype = dtype, count=rec_len)
                yield(data_to_return)


    ##-----------------
    ##-----------------
    # GET THE INDICATORS FOR `STICK', `SLIP' AND `UNKNOWN'
    # FROM A DATA FILE SPECIFIED BY USER
    ##
    def get_indicators_from_file(self, filename, **kwargs):

        with open(filename, "r") as slip_f:
            lines = slip_f.readlines()

        for line_id in range(len(lines)):
            time, ind = lines[line_id].split()
            if (time >= self.min_record) and (time < self.max_record):
                self.indicators_for_data.append(int(ind))
            if time > self.max_record:
                break
        slip_f.close()

    ##-----------------
    ##-----------------
    # GET THE INDICATORS FOR `STICK', `SLIP' AND `UNKNOWN'
    # FROM COMPUTATIONS
    ##

    def get_indicators(self, average_over, prefix_interval, wall_length, threshold_size):

        self.average_over = average_over
        self.prefix_interval = prefix_interval
        self.wall_length = wall_length
        self.threshold_size = threshold_size

        wall_pos = []
        for record in self.data:
            wall_pos.append(get_wall_position(record))


        ##-----------------
        # assume no slipping first and check for a possible slip -> the ax_1 position needs to change
        # by at least threshold_size to be considered as a possible slip
        ##-----------------
        slipping = False

        for start_ in range(prefix_interval['ax_1'] + 1,len(lines)-1-2*postfix_interval['ax_1']):
            current_ax_1 = wall_pos['ax_1'][start_]
            postfix_ax_1= wall_pos['ax_1'][start_+postfix_interval['ax_1']]

            indicator = 0

            if abs(current_ax_1 - postfix_ax_1)%wall_length > threshold_size:
                indicator += 1

                ##-----------------
                # if the ax_1 moves by at least threshold_size, check the variation in ax_2 movement
                ##-----------------
                if not slipping:
                    std_postfix_interval = np.std(wall_pos['ax_2'][start_ : start_ + postfix_interval['ax_2']])
                    std_prefix_interval = np.std(wall_pos['ax_2'][start_ - prefix_interval['ax_2'] : start_])

                    ##-----------------
                    # if the standard deviation of the next interval increases, the there is a slip
                    ##-----------------
                    if std_postfix_interval > std_prefix_interval:
                        slipping = True

        ##-----------------
        ##
        ##-----------------

            if slipping:
                if indicator:
                    indicator += 1
                else:
                    slipping = False

            self.indicators_for_data.append(indicator)

        return
