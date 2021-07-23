#### Brief

This is a multivariate-normal density probability function implementation by OpenCV, which can achieve the same result like **scipy.stats.multivariate_normal.pdf()** in Python.



#### Usage

```c++
MultiVariateGaussian MVG;
cv::Mat train_data(10, 3, CV_32FC1);
cv::Mat val_data(5, 3, CV_32FC1);

// random data
cv::RNG rnger(cv::getTickCount());
rnger.fill(train_data, cv::RNG::UNIFORM, cv::Scalar::all(0), cv::Scalar::all(255));
rnger.fill(val_data, cv::RNG::UNIFORM, cv::Scalar::all(0), cv::Scalar::all(255));
    
// estimate gaussian 
MVG.estimate_gaussian(train_data);
cv::Mat val_data_prob = MVG.multivariate_gaussian(val_data);
```



#### Output Example

```
data: [31.365021, 238.91919, 29.257645] dense probability: [8.242923853648458e-09]
data: [232.97113, 57.590729, 203.72713] dense probability: [3.033565617629424e-08]
data: [98.62838, 212.40479, 241.48314]  dense probability: [4.517634241939982e-08]
data: [103.37743, 58.68898, 96.820831]  dense probability: [2.001677126583222e-08]
data: [151.7095, 151.72896, 250.50418]  dense probability: [3.933165200360655e-08]
```

