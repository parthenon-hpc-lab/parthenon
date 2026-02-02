### TODO
  - [ ] Decide if we want to provide QR decomposition (beyond the simple version we have for testing)
  - [x] Move `Tridiagonalize` and `ImplicitQR` to a class that allocates scratch storage
  - [x] Test things by wrapping in `par_for_outer`
  - [x] Switch to `par_for_inner` loops where possible
  - [x] Fix the tests (may just be too stringent)
  - [x] Merge `par_for_refactor` branch back into main
  - [x] Compute eigen vectors
  - [x] Test eigenvector calculations
  - [x] Add Golub-Kahan SVD
  - [x] Add tests for Golub-Kahan SVD
  - [x] Determine why the `CheckSingularValueSanity` part of the test ChatGPT wrote is failing. All 
        tests succeed if this one requirement is commented out. [Is this just a convention thing and 
        we should be multiplying through one of the columns of U or V? It is just a sign convention, so removed the test.]
  - Repo cleanup:
    - [ ] Move loop stuff out of matrix.hpp
    - [ ] Create impl/ directory
    - [ ] Move matrix.hpp/cpp to tests
- [ ] Try Gram-SVD