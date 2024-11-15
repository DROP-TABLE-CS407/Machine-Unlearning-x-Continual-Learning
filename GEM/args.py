class Args:
    def __init__(self):
        # model parameters
        self.model = 'single'
        self.n_hiddens = 10
        self.n_layers = 2

        # memory parameters
        self.n_memories = 256
        self.memory_strength = 0.5
        self.finetune = 'no'

        # optimizer parameters
        self.n_epochs = 15
        self.batch_size = 10
        self.lr = 0.1

        # experiment parameters
        self.cuda = 'yes'
        self.seed = 0
        self.log_every = 1
        self.save_path = 'results/'

        # data parameters
        self.data_path = 'data/'
        self.data_file = 'mnist_permutations.pt'
        self.samples_per_task = 2500
        self.shuffle_tasks = 'no'

        # Convert string flags to boolean
        self.cuda = True if self.cuda == 'yes' else False
        self.finetune = True if self.finetune == 'yes' else False