import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from distfit import distfit
from scipy import stats
from scipy.stats import gamma

# Read the CSV file into a pandas DataFrame
yu = pd.read_csv("data.csv")
# Display the DataFrame
# k = []
# for i in yu["x"]:
#   k.append(float(i))

# data = np.array(k)
data = np.array(yu["x"].astype(float))
n = data.shape[0]

# Plot histogram
plt.hist(data, bins=30, color='skyblue', edgecolor='black', density=True, alpha=0.7, label='Histogram')
# # Plot histogram
# plt.hist(data, bins=80, color='skyblue', edgecolor='black')
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

ans_to_hold = scipy.stats.gamma.fit(data)
print(ans_to_hold)

print("Location parameter (theta):", ans_to_hold[1])
print("Shape parameter (alpha):", ans_to_hold[0])
print("Scale parameter (beta):", ans_to_hold[2])
x = np.linspace(np.min(data),np.max(data)+1)

# r = scipy.stats.expon.fit(data)[0]
# print(r)
plt.plot(x, scipy.stats.gamma.pdf(x, a=ans_to_hold[0], loc = ans_to_hold[1], scale=ans_to_hold[2]), color='red', label='Gamma PDF')
# exponential_pdf = scipy.stats.expon.pdf(x, scale=r)

# plt.plot(x, exponential_pdf, color='purple', label='Exponential PDF')
# plt.show()
plt.legend()

# # graph dekh ke lg rha hai ki gamma hoga

ans_to_hold = scipy.stats.gamma.fit(data)
print(ans_to_hold)
print("Location parameter (theta):", ans_to_hold[1])
print("Shape parameter (alpha):", ans_to_hold[0])
print("Scale parameter (beta):", ans_to_hold[2])

# # yeh hm inbuilt libnrary se bhi kr skte hai but uske bina bhi ho jayega




# Estimate location parameter (theta)
theta = np.min(data) # minimum hota hai

# Adjust the data by subtracting theta
adjusted_data = data

# Calculate sample mean and variance of adjusted data
mu = np.mean(adjusted_data)
sigma_squared = np.var(adjusted_data)

# Estimate shape parameter (alpha)
alpha = (mu ** 2) / sigma_squared

# Estimate scale parameter (beta)
beta = (sigma_squared / mu)
# Print the estimated parameters
print("Location parameter (theta):", theta)
print("Shape parameter (alpha):", alpha)
print("Scale parameter (beta):", beta)


# trying exponential also

lambda_ = 1 / np.mean(data)

print("Exponential check also Lamba = ",lambda_) # lambda reserved keyword hota hai


print("Mean of the data",np.mean(data))
print("Variance of the data",np.var(data))


plt.hist(data, bins=30, color='skyblue', edgecolor='black', density=True, label='Histogram')
# # Plot histogram
# plt.hist(data, bins=80, color='skyblue', edgecolor='black')
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
# plt.show()

x = np.linspace(min(data), max(data)+1)
plt.plot(x, scipy.stats.gamma.pdf(x, a=alpha, loc=theta, scale=beta), color='red', label='Gamma PDF')



# Generate x values for the PDF plot
# x = np.linspace(0, max(data), 1000)

# Calculate the exponential PDF values using the rate parameter
exponential_pdf = scipy.stats.expon.pdf(x, scale=1/lambda_)
normal_pdf = scipy.stats.norm.pdf(x, loc=mu, scale=np.sqrt(sigma_squared))
# Plot the exponential PDF
plt.plot(x, exponential_pdf, color='purple', label='Exponential PDF')
plt.plot(x, normal_pdf, color='green', label='Normal Pdf')
plt.legend()

plt.show()
print("BEST FIT ESTIMATE IS GAMMA FUNCTION")

# Question 2 parameter calculation


# Estimate location parameter (theta)
theta = np.min(data) # minimum hota hai

# Adjust the data by subtracting theta
adjusted_data = data

# Calculate sample mean and variance of adjusted data
mu = np.mean(adjusted_data)
sigma_squared = np.var(adjusted_data)

# Estimate shape parameter (alpha)
alpha = (mu ** 2) / sigma_squared

# Estimate scale parameter (beta)
beta = (sigma_squared / mu)
# Print the estimated parameters
print("Location parameter (theta):", theta)
print("Shape parameter (alpha):", alpha)
print("Scale parameter (beta):", beta)


# trying exponential also

lambda_ = 1 / np.mean(data)

print("Exponential check also Lamba = ",lambda_) # lambda reserved keyword hota hai

print('''
    I have shown on my copy proved in short to show that these estimators are unbiased efficient and consistent also for values i have calculated already above Please Note X(bar) is also a unbiased estimator for it X(bar) = alpha*beta.
    Question 3 
        a) for alpha found 2 estimators unbiased and consistency is one other is biased 
        b) for beta found 2 estimators unbiased and consistency is one other is biased 
''')


# Question 2 and 4

print("Found on Notebook values are \n alpha = X(bar)**2/((1/n)*(summ(Xi - X(bar))**2)) ")
print("Found on Notebook values are \n alpha = 2*X(bar)**2/((1/n)*(summ(Xi - X(bar))**2)) ")

print("Beta = (1/n)*summ(Xi - X(bar))**2/X(bar)")

print("Beta = 2*(1/n)*summ(Xi - X(bar))**2/X(bar)")



# Question 5 to find UMVUE

print("UMVUE for beta is ", beta**2/((alpha)*n))
# print(np.var(np.mean(data)))
print("Same estimates will be found","minimum variance is",beta**2/((alpha)*n))

print("UMVUE for alpha is ", 2*alpha**2/(n*(2*alpha+1)))
print("Some estimator will be found",alpha/n)

# Question 5 spl 
print(2*alpha*alpha/(n*(2*alpha+1))) #minimum achievable variance 

print(alpha/n) # hence we cannot say that it will be efficient





# Question 6
print("Question 6 Interval Estimation")

# Recalculate alpha and beta using method of moments
mean_x = data.mean()
variance_x = data.var()

alpha_hat = mean_x**2 / variance_x
beta_hat = mean_x / alpha_hat


se_alpha = alpha_hat / np.sqrt(data.shape[0])


confidence_levels = [0.01, 0.05, 0.1]
z_values = {conf: stats.norm.ppf(1 - conf/2) for conf in confidence_levels}


ci = {conf: (alpha_hat - z * se_alpha, alpha_hat + z * se_alpha) for conf, z in z_values.items()}

print("alpha hat is ",alpha_hat,"beta hat is ",beta_hat)

for i in ci:
    print(i,":",ci[i])

import math

from scipy.stats import t

n = 1000  # Sample size
s_squared = 0.243  # Sample variance
mean_0 = 0.525  # Hypothesized variance
real_mn = 0.515

print("H0 : mean = 0.525")
print("H0 : mean ≠ 0.525")

# Calculating the t-test statistic
t_statistic = (real_mn - mean_0) / (math.sqrt(s_squared / n))

# Degrees of freedom
df = n - 1

# Significance level for two-tailed test
alpha = 0.05

# Critical values for both tails
lower_critical_value = t.ppf(alpha / 2, df)  # Lower tail
upper_critical_value = t.ppf(1 - alpha / 2, df)  # Upper tail

print(lower_critical_value, upper_critical_value)  # hypothesis test
print(t_statistic)
# Decision based on the critical values
print("Failed to reject the null hypothesis") if t_statistic > lower_critical_value and t_statistic < upper_critical_value else print("Failed to accept the null hypothesis")


# Given values
n = 1000  # Sample size
s_squared = 0.24  # Sample variance
sigma_squared_0 = 0.25  # Hypothesized variance

print("H0 : s_squared = 0.24")
print("H0 : s_squared ≠ 0.24")
# Calculating the chi-squared test statistic
chi_squared_statistic = (n - 1) * s_squared / sigma_squared_0
print(chi_squared_statistic)

from scipy.stats import chi2

# Significance level for two-tailed test
alpha = 0.05
df = n - 1  # Degrees of freedom

# Critical values for both tails
lower_critical_value = chi2.ppf(alpha / 2, df)  # Lower tail
upper_critical_value = chi2.ppf(1 - alpha / 2, df)  # Upper tail

print(lower_critical_value, upper_critical_value)  # hypothesis test 

print("Failed to reject the null hypothesis") if chi_squared_statistic > lower_critical_value and chi_squared_statistic < upper_critical_value else print("Failed to accept the null hypothesis")



print("Goodness of test")

test_stat = "chi_squared_statistic"
alpha = 0.05
print("start kre")
rep = 0
mu = np.mean(data)
for i in data:
    rep +=(i-mu)**2
rep/=mu

# print(rep)

ans = chi2.ppf(1-alpha, df)
print("my calculated answer is ",rep)
print("table value is ",ans)
print("Failed to reject the null hypothesis") if rep < ans else print("Failed to accept the null hypothesis")


