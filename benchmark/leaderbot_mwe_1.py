# Install ;#\package#; with ;#"\texttt{pip install \package}"#;
import leaderbot as lb

# Load default dataset shipped with the package
data = lb.data.load()

# Create Davidson model with covariance factor ;#\mbox{$k_{\mathrm{cov}}=0$}#; (diagonal covariance)
# and tie factor ;#$k_{\mathrm{tie}}=0$#;. This corresponds to Model 23 in ;# \Cref{tab:model-selection} #;
model = lb.models.DavidsonFactor(data, n_cov_factors=0, n_tie_factors=0)

# Train the model
model.train(method='BFGS', max_iter=1500, tol=1e-8)

# Make inference
probabilities = model.infer(data)

# Make prediction
preiction = model.predict(data)

# Compute loss function ;#$ -\ell(\vect{\theta}) $#; and its Jacobian ;#$ -\partial \ell(\vect{\theta}) / \partial \vect{\theta} $#;
loss, jac = model.loss(return_jac=True)

# Print leaderboard and plots overall probabilities
model.leaderboard(max_rank=None, plot=True)

# Generates ;# \Cref{fig:scores} #;
model.plot_scores(max_rank=50)

# Rank competitors based on their scores
rank = model.rank()

# Visualize correlation similar to ;# \Cref{fig:kpca} #; using Kernel PCA method
# projected on 3-dimensional space for the top 40 ranks.
model.visualize(max_rank=40, method='kpca', dim='3d')

# Generate a plot similar to ;# \Cref{fig:match-matrix} #; with the win/loss matrix ;#$ \tens{W} $#; and
# tie matrix ;#$ \tens{T} $#; for both observed and predicted probabilities.
model.match_matrix(max_rank=25, win_range=[0.2, 0.6], tie_range=[0.15, 0.4])