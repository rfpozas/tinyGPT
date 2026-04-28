from dataclasses import asdict, dataclass

@dataclass(slots=True)
class TrainConfig:
	batch_size: int = 32 # number of sequences processed in parallel during training
	max_iters: int = 5000 # total number of training iterations (batches) to perform
	eval_interval: int = 250 # how often to evaluate the model on train and val sets (in number of iterations)
	eval_iters: int = 20 # number of batches to use for evaluation (higher is more accurate, but slower)
	learning_rate: float = 1e-3 # learning rate for the optimizer
	weight_decay: float = 0.01 # weight decay for regularization
	split_ratio: float = 0.9 # ratio of data to use for training vs validation (between 0 and 1)
	seed: int = 0 # random seed for reproducibility
	checkpoint_path: str = "artifacts/tinygpt.pt" # path to save the trained model checkpoint
	history_path: str = "artifacts/train_history.json" # path to save the training history (losses over time)

	def __post_init__(self) -> None:
		if self.batch_size <= 0:
			raise ValueError("batch_size must be positive")
		if self.max_iters <= 0:
			raise ValueError("max_iters must be positive")
		if self.eval_interval <= 0:
			raise ValueError("eval_interval must be positive")
		if self.eval_iters <= 0:
			raise ValueError("eval_iters must be positive")
		if not 0 < self.split_ratio < 1:
			raise ValueError("split_ratio must be between 0 and 1")
		if self.learning_rate <= 0:
			raise ValueError("learning_rate must be positive")
		if self.weight_decay < 0:
			raise ValueError("weight_decay must be non-negative")

	def to_dict(self) -> dict[str, int | float | str]:
		return asdict(self)
