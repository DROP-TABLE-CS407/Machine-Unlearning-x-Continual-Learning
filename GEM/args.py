class Args:
    def __init__(self):
        # model parameters
        self.model = 'single'
        self.n_hiddens = 100
        self.n_layers = 2

        # memory parameters
        self.n_memories = 0
        self.memory_strength = 0.0
        self.finetune = 'no'

        # optimizer parameters
        self.n_epochs = 1
        self.batch_size = 10
        self.lr = 1e-3

        # experiment parameters
        self.cuda = 'no'
        self.seed = 0
        self.log_every = 100
        self.save_path = 'results/'

        # data parameters
        self.data_path = 'data/'
        self.data_file = 'mnist_permutations.pt'
        self.samples_per_task = -1
        self.shuffle_tasks = 'no'

        # Convert string flags to boolean
        self.cuda = True if self.cuda == 'yes' else False
        self.finetune = True if self.finetune == 'yes' else False