class Args:
    def __init__(self, learning_rate = 0.1):
        # model parameters
        self.model = 'single'
        self.n_hiddens = 10
        self.n_layers = 2

        # memory parameters
        self.n_memories = 256
        self.memory_strength = 0.5
        self.finetune = 'no'
        self.mem_cnt = 5120

        # optimizer parameters
        self.n_epochs = 3
        self.batch_size = 10
        self.lr = learning_rate
        
        self.unlearn_batch_size = 16
        self.unlearn_epochs = 5
        self.unlearning_rate = 0.01
        self.unlearn_mem_strength = 0.5

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