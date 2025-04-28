class Args:
    def __init__(self, learning_rate = 0.1):
        # model parameters
        self.model = 'single'
        self.n_hiddens = 10
        self.n_layers = 2

        # memory parameters
        self.n_memories = 256
        self.n_learn_memories = 256
        self.n_unlearn_memories = 256
        self.memory_strength = 0.5
        self.finetune = 'no'
        self.mem_cnt = 5120

        # optimizer parameters
        self.n_epochs = 3
        self.batch_size = 10
        self.lr = learning_rate
        
        self.unlearn_batch_size = 8
        self.unlearn_epochs = 48
        self.unlearning_rate = 0.01
        self.unlearn_mem_strength = 0.5
        
        # memorization buffers
        self.mem_learning_buffer = True
        self.learning_buffer_split = 0.5
        self.learning_buffer_type = "least"

        self.mem_unlearning_buffer = True
        self.unlearning_buffer_split = 0.5
        self.unlearning_buffer_type = "most"
        
        self.algorithm = 'neggem'
        self.alpha = 0.5
        
        # for rum
        self.use_rum = False
        self.rum_split = 0.5
        self.rum_memorization = "a"

        # experiment parameters nor do we use this stuff 
        self.cuda = 'yes'
        self.seed = 69420
        self.log_every = 1
        self.save_path = 'results/'

        # data parameters -- we dont use this stuff
        self.data_path = 'data/'
        self.data_file = 'mnist_permutations.pt'
        self.samples_per_task = 2500
        self.shuffle_tasks = 'no'

        # Convert string flags to boolean
        self.cuda = True if self.cuda == 'yes' else False
        self.finetune = True if self.finetune == 'yes' else False

        self.salun = False
        self.salun_threshold = 0.95
