# Explainability report: GunPoint

* Test accuracy: **99.33%**
* Feature set chosen from training data: **PPV**
* Training examples: 50, test examples: 150, series length: 150

## What separates the classes

* Class 1: strongest evidence at 12%-22% of the series (t = 18-32). There the series is on average LOWER than in other classes (-0.76 vs -0.49) and rising (slope 0.016 vs 0.076).
* Class 2: strongest evidence at 12%-22% of the series (t = 18-32). There the series is on average HIGHER than in other classes (-0.49 vs -0.76) and rising (slope 0.076 vs 0.016).

## Most important dilations (pattern widths)

* dilation 1: kernel spans 9 time steps, 23.4% of total importance
* dilation 2: kernel spans 17 time steps, 14.3% of total importance
* dilation 3: kernel spans 25 time steps, 11.6% of total importance
* dilation 4: kernel spans 33 time steps, 10.5% of total importance
* dilation 7: kernel spans 57 time steps, 6.6% of total importance

## Most important kernels

* kernel 41 weights [-1, 2, -1, -1, 2, -1, -1, 2, -1]: 2.2%
* kernel 43 weights [-1, 2, -1, -1, -1, 2, 2, -1, -1]: 1.6%
* kernel 25 weights [2, -1, -1, -1, -1, -1, 2, 2, -1]: 1.6%
* kernel 23 weights [2, -1, -1, -1, -1, 2, -1, 2, -1]: 1.6%
* kernel 24 weights [2, -1, -1, -1, -1, 2, -1, -1, 2]: 1.5%
