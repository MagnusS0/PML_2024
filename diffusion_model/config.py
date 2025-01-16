class Config:
    def __init__(self, T=1000, learning_rate=1e-3, epochs=100, batch_size=256, 
                 model_type="DDPM", beta_schedule="cosine", loss_type="IS",
                 eval_fid_samples=10000, eval_is_samples=1024, eval_is_splits=5):
        self.T = T
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_type = model_type
        self.beta_schedule = beta_schedule
        self.loss_type = loss_type
        self.eval_fid_samples = eval_fid_samples
        self.eval_is_samples = eval_is_samples
        self.eval_is_splits = eval_is_splits

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

# Default configuration
default_config = {
    "T": 1000,
    "learning_rate": 1e-3,
    "epochs": 100,
    "batch_size": 256,
    "model_type": "DDPM",
    "beta_schedule": "cosine",
    "loss_type": "IS",
    "eval_fid_samples": 10000,
    "eval_is_samples": 1024,
    "eval_is_splits": 5
}
