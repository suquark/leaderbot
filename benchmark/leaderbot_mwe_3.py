import leaderbot as lb

# Load dataset
data = lb.data.load()

# Split data into training and test sets
training_data, test_data = lb.data.split(data, test_ratio=0.1, seed=20)

# List models 1, 2, 11, 12, 23, and 24 from ;# \Cref{tab:model-selection} #;
models = [
    lb.models.BradleyTerry(training_data, k_cov=None),
    lb.models.BradleyTerry(training_data, k_cov=0),
    lb.models.RaoKupper(training_data, k_cov=0, k_tie=0),
    lb.models.RaoKupper(training_data, k_cov=0, k_tie=1),
    lb.models.Davidson(training_data, k_cov=0, k_tie=0),
    lb.models.Davidson(training_data, k_cov=0, k_tie=1)]

# Evaluate models for generalization on test data, similar to ;# \Cref{tab:generalization} #;
gen_metrics = lb.evaluate.generalization(models, test_data=test_data,
                                                      train=True, metric='RMSE', report=True)
