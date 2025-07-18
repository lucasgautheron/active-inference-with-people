# Adaptive Bayesian testing with PsyNet

This experiment implements real-time adaptive Bayesian testing using PsyNet and Variational Bayesian Optimal
Experimental Design.

The goal of the experiment is to infer the general knowledge of participants using 45 trivia questions (the questions
were taken from Dubourg et al., 2025).
The ability of the participants and the difficulty of the questions are learned continuously using a Bayesian
item-response model.
The optimal challenge to present to each participant is evaluated at every trial by maximizing the expected information
gain using variational inference methods for efficient computation.

# PsyNet

This experiment is implemented using the [PsyNet framework](https://www.psynet.dev/).

For installation instructions, see docs/INSTALL.md.

For a list of run commands, see docs/RUN.md.

For more information about PsyNet, see the [documentation website](https://psynetdev.gitlab.io/PsyNet/).

# References

* Foster, A., Jankowiak, M., Bingham, E., Horsfall, P., Teh, Y. W., Rainforth, T., & Goodman, N. (2019). Variational
  Bayesian optimal experimental design. Advances in neural information processing systems, 32.
* Dubourg, E., Dheilly, T., Mercier, H., & Morin, O. (2025). Using the Nested Structure of Knowledge to Infer What
  Others Know. Psychological Science, 36(6), 443-450. https://doi.org/10.1177/09567976251339633 (Original work published
  2025)