# Install ;#\package#; with ;#"\texttt{pip install \package}"#;
import leaderbot as lb

# Load default dataset shipped with the package
data = lb.data.load()

# Split data to training and test
training_data, test_data = lb.data.split(data, test_ratio=0.1)

# Create a statistical model based on the Davidson method
model = lb.models.DavidsonScaled(training_data) #, n_tie_factors=0)

# Train the model
model.train(method='BFGS', max_iter=1500, tol=1e-8)

# Make inference and prediction
probabilities = model.infer(test_data)

# Make prediction on test data
preiction = model.predict(test_data)

# If needed, compute loss function ;#% \ell(\vect{theta}) $#; and
# its Jacobian ;#$ \partial \ell / \partial \vect{\theta}} $#;
loss, jac = model.loss(return_jac=True, constraint=False)

# Print leaderboard and plots overall probabilities
model.leaderboard(max_rank=50, plot=True)

# Generates ;# \Cref[fig:scores] #;
model.plot_scores(max_rank=50)

# Returns ranks
model.rank()

# Visualize correlation similar to ;# \Cref{fig:visualization} #;
# using Kernel PCA method projected on 3-dimensional space for
# the top 40 ranks.
model.visualize(max_rank=40, method='kpca', dim='3d')
model.plot_scores(max_rank=50)

# Generate plot similar to ;# \Cref{fig:match-matrix} #; with the
# win/loss matrix ;#$ \tens[{W} $#; and tie matrix
# ;#$ \tens{T} $#; for both observed and predicted probabilities.
model.match_matrix(max_rank=25, density=True, source='both',
                   win_range=[0.2, 0.6], tie_range=[0.25, 0.4])
