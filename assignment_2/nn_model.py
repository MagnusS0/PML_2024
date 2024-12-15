import pyro
import numpy
import torch
import sklearn
import sklearn.datasets
import sklearn.model_selection
import matplotlib.pyplot as plt
import pyro.distributions as pdist
import arviz
from pyro.infer import MCMC, NUTS
import pandas as pd

seed_value = 42
torch.manual_seed(seed_value)
pyro.set_rng_seed(seed_value)
numpy.random.seed(seed_value)

def accuracy(pred, data):
  """
  Calculate accuracy of predicted labels (integers).

  pred: predictions, tensor[sample_index, chain_index, data_index, logits]
  data: actual data (digit), tensor[data_index]

  Prediction is taken as most common predicted value.
  Returns accuracy (#correct/#total).
  """
  n=data.shape[0]
  correct=0
  total=0
  for i in range(0, n):
      # Get most common prediction value from logits
      pred_i=int(torch.argmax(torch.sum(pred[:,0,i,:],0)))
      # Compare prediction with data
      if int(data[i])==int(pred_i):
          correct+=1.0
      total+=1.0
  # Return fractional accuracy
  return correct/total

# Iris data set
Dx=4 # Input vector dim
Dy=3 # Number of labels

iris=sklearn.datasets.load_iris()
x_all=torch.tensor(iris.data, dtype=torch.float) # Input vector (4D)
y_all=torch.tensor(iris.target, dtype=torch.int) # Label(3 classes)

# Make training and test set
x, x_test, y, y_test = sklearn.model_selection.train_test_split(
    x_all, y_all, test_size=0.33, random_state=42)

print("Data set / test set sizes: %i, %i." % (x.shape[0], x_test.shape[0]))

class Model(pyro.nn.PyroModule):
    def __init__(self, x_dim=4, y_dim=3, h_dim=5):
        self.x_dim=x_dim
        self.y_dim=y_dim
        self.h_dim=h_dim
        super().__init__()

    def __call__(self, x, y=None):
        """
        We need None for predictive
        """
        x_dim=self.x_dim
        y_dim=self.y_dim
        h_dim=self.h_dim
        # Number of observations
        n=x.shape[0]
        # standard deviation of Normals
        sd=1 # EXERCISE: 100->1
        # Layer 1
        w1=pyro.sample("w1", pdist.Normal(0, sd).expand([x_dim, h_dim]).to_event(2))
        b1=pyro.sample("b1", pdist.Normal(0, sd).expand([h_dim]).to_event(1))
        # Layer 2 # EXERCISE: added layer
        w2=pyro.sample("w2", pdist.Normal(0, sd).expand([h_dim, h_dim]).to_event(2))
        b2=pyro.sample("b2", pdist.Normal(0, sd).expand([h_dim]).to_event(1))
        # Layer 3
        w3=pyro.sample("w3", pdist.Normal(0, sd).expand([h_dim, y_dim]).to_event(2))
        b3=pyro.sample("b3", pdist.Normal(0, sd).expand([y_dim]).to_event(1))
        # NN
        h1=torch.tanh((x @ w1) + b1)
        h2=torch.tanh((h1 @ w2) + b2) # EXERCISE: added layer

        logits=(h2 @ w3 + b3)
        # Save deterministc variable (logits) in trace
        pyro.deterministic("logits", logits)
        # Categorical likelihood
        with pyro.plate("labels", n):
            obs=pyro.sample("obs", pdist.Categorical(logits=logits), obs=y)
        

def run_inference(
    model,
    x=None,
    y=None,
    num_warmup=1000,
    num_samples=1000,
    max_tree_depth=10,
    dense_mass=False
):
    if x is None or y is None:
        raise ValueError("Both x and y must be provided")

    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.long)

    pyro.clear_param_store()

    try:
        kernel = NUTS(
            model, 
            full_mass=dense_mass, 
            max_tree_depth=max_tree_depth, 
            jit_compile=True
        )
        mcmc = MCMC(
            kernel,
            num_samples=num_samples,
            warmup_steps=num_warmup,
            num_chains=2,
            mp_context="spawn",
        )

        mcmc.run(x, y)
        return mcmc

    except Exception as e:
        print(f"MCMC inference failed: {str(e)}")
        raise

if __name__ == "__main__":
    S = 1000
    model = Model()

    try:
        mcmc = run_inference(
            model, 
            x=x,
            y=y, 
            num_warmup=1000, 
            num_samples=S, 
            max_tree_depth=10, 
            dense_mass=False
        )

        data = arviz.from_pyro(mcmc, log_likelihood=False)
        summary = arviz.summary(data, round_to=2)
        print("\nModel Summary:")
        print(summary)

        pd.DataFrame(summary).to_latex("summary.tex")

        arviz.plot_trace(data)
        plt.savefig('trace_plot.png')
        plt.show()

        # Predictive
        predictive = pyro.infer.Predictive(model, mcmc.get_samples())
        pred = predictive(x_test)
        acc = accuracy(pred["logits"], y_test)
        print(f"Accuracy: {acc:.4f}")

    except Exception as e:
        print(f" Model failed: {str(e)}")
        raise