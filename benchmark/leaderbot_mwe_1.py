# Install ;#\package#; with ;#"\texttt{pip install \package}"#;
import leaderbot as lb

# Load the default dataset shipped with the package
data = lb.data.load()

# Create a Davidson model with covariance factor ;#\mbox{$k_{\mathrm{cov}}=0$}#; (diagonal covariance)
# and tie factor ;#$k_{\mathrm{tie}}=0$#;. This corresponds to Model 23 in ;# \Cref{tab:model-selection}#;
model = lb.models.Davidson(data, k_cov=0, k_tie=0)

# Train the model
model.train(method='BFGS', max_iter=1500, tol=1e-8)

# Make inference and prediction
probabilities = model.infer(data)
prediction = model.predict(data)

# Compute the loss function ;#$-\ell(\vect{\theta})$#;, its Jacobian ;#$-\partial \ell(\vect{\theta}) / \partial \vect{\theta}$#;, and Hessian ;#$-\nabla_{\vect{\theta}} \nabla_{\vect{\theta}}^{\intercal} \ell(\vect{\theta})$#;
loss, jac = model.loss(return_jac=True)
hess = model.fisher()

# Plot marginal probabilities (similar to ;# \Cref{fig:prediction-error}#;)
model.marginal_outcomes()

# Generate a plot for competitor scores (similar to ;# \Cref{fig:scores}#;)
model.plot_scores(max_rank=50)

# Rank competitors based on their scores, print leaderboard
rank = model.rank()
model.leaderboard()

# Visualize correlation using Kernel PCA projected in 3D space (similar to ;# \Cref{fig:kpca}#;)
# Use ;#\texttt{method='mds'}#; to generate an MDS plot (similar to ;# \Cref{fig:mds}#;)
model.map_distance(max_rank=40, method='kpca', dim='3d')

# Generate a match matrix plot for observed and predicted win/loss probabilities
# and tie probabilities (similar to ;# \Cref{fig:match-matrix}#;)
model.match_matrix(max_rank=25, win_range=[0.2, 0.6], tie_range=[0.15, 0.4])

# Perform hierarchical clustering for the top 100 competitors (similar to ;# \Cref{fig:cluster}#;)
model.cluster(max_rank=100)
