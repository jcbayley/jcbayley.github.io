---
title: 'Bayesian fitting with emcee'
date: 2012-08-14
permalink: /posts/2019/bayesian-fitting
tags:
  - bayesian
  - emcee
---

```python
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit,minimize
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy
import emcee
import corner
import time
```

# Simple Bayesian parameter estimation

Before any analysis starts there are genreally three main elements:

1. data ${\boldsymbol d}$, this is the data which you have collected
2. a model $I$ which has some parameters ${\boldsymbol \theta}$
3. some information about the model you have prior to seeing any data, i.e. expected values of parameters


What we aim to find is the the probability distribution of the model parameters under the assumption that the model is correct and some observation of data, i.e. $p(\theta | d, I)$. This probability distribution contains all information about the parameters given the data, including the most likely value and the uncertainty associated with it. 

This gives us an idea of all set of parameters which are consistent with our data rather than just the best fit set of parameters, giving us a complete view of the problem.



## Basic probability

Initially I will define some basic concepts of probability.  We can define the
probability of some event $A$ as $p(A)$ where probabilities lie in the range $0 \leq p(A)
\leq 1$ and some other event $B$ which has a probability $p(B)$ and
which lies in the range $0 \leq p(B) \leq 1$.

### Union
A union is the probability of either event $A$ happening or event $B$ happening. This is written as, $p(A \cup B)$.
	
### Intersection
An intersection is then the probability that both an event $A$ and an event $B$ happens. This is written as $p(A \cap B)$.

### Independent and dependent Events
If the event $A$ is dependent on event $B$, i.e. the event $A$ affects event $B$ or vice versa, then the joint probability of both events is
\begin{equation}
p(A \cap B) = p(A)p(B \mid A) = p(B)p(A \mid B).
\end{equation}
Here $p(B \mid A)$ means the probability of event $B$ given an event $A$.
However, if the events $A$ and $B$ are independent, i.e. the event $A$ does not affect the outcome of event $B$, then
\begin{equation}
p(A \cap B) = p(A)p(B).
\end{equation}


### Conditional probability
Conditional probability arises from situations where one event $A$ affects the event $B$.
The definition of this arises from the dependent events defined above 
\begin{equation}
p(A \mid B) = \frac{p(A \cap B)}{p(B)}.
\end{equation}

### Bayes Theorem
Bayes theorem can then be defined using conditional probabilities. i.e we can use
\begin{equation}
p(A \mid B) = \frac{p(A \cap B)}{p(B)} \quad \rm{and} \quad p(B \mid A) = \frac{p(A \cap B)}{p(A)}
\end{equation}
such that
\begin{equation}
p(B)p(A \mid B) = p(A)p(B \mid A)
\end{equation}
and this is rearranged to Bayes theorem
\begin{equation}
p(A \mid B) = \frac{p(A)p(B \mid A)}{p(B)}
\end{equation}



We can take Bayes theorem from above and apply it to a
problem which involves inferring the parameters from some model. Here we can
relabel the events $A$ and $B$ with the data ${\boldsymbol d}$ and the parameters ${\boldsymbol
\theta}$ of some model $I$.  Bayes theorem then becomes

\begin{equation}
p({\boldsymbol \theta} \mid {\boldsymbol d}, I) = \frac{p({\boldsymbol \theta}, I)p({\boldsymbol d} \mid {\boldsymbol \theta}, I)}{p({\boldsymbol d} \mid I)}
\end{equation}

where each of the components are assigned names: $p({\boldsymbol \theta} \mid {\boldsymbol d})$
is the posterior distribution, $p({\boldsymbol \theta})$ is the prior distribution,
$p({\boldsymbol d} \mid {\boldsymbol \theta})$ is the likelihood, and $p({\boldsymbol d})$ is the
Bayesian Evidence.

### Posterior
The posterior distribution describes the probability of a parameter
${\boldsymbol \theta}$ in some model $I$ given some data $\boldsymbol{d}$. For many problems this is
the distribution which is most useful as it informs you how likely any set of
parameters from your model are given some observation.
	
### Prior
The Prior distribution is a key part of Bayesian
statistics which describes the distribution of the parameters ${\boldsymbol \theta}$ given the model $I$. This should reflect any beliefs about the parameters $\boldsymbol{\theta}$ prior to the observations.
	
### Likelihood
The likelihood is where the observation is included
in the calculation. This tells you how probable it is to get the observed data
$\boldsymbol{d}$ given the model $I$ with the set of parameters $\boldsymbol{\theta}$. 
	
### Bayesian Evidence
This is the probability of the data itself
given the choice of model. This is found by integrating the likelihood over all
possible values of ${\boldsymbol \theta}$ weighting them by our prior belief of that
value of ${\boldsymbol \theta}$. This is also known as the marginal
likelihood and is defined by,

\begin{equation}
    p({\boldsymbol d} \mid I) = \int p({\boldsymbol \theta}, I)p({\boldsymbol d} \mid {\boldsymbol \theta}, I) d{\boldsymbol \theta}.
\end{equation} 



## How do we use this
Initially to simplify the problem we will assume our model is correct we can write the posterior likelihood as:
\begin{equation}
p({\boldsymbol \theta} \mid {\boldsymbol d}) \propto p({\boldsymbol \theta})p({\boldsymbol d} \mid {\boldsymbol \theta}).
\end{equation}
</br>
The next step is to mathematically write out the prior distribution and posterior distribution. For simplicity we will assume that our prior distribution is flat, i.e. we know nothing about the parameters of the model.
To write the likelihood we have to think about our data. We can define each data point as some background noise plus the signal we are interested in i.e.:
\begin{equation}
d_i = y(x_i,{\boldsymbol \theta}) + \mathcal{N}(0, \sigma),
\end{equation}
where $d_i$ is our measured data point, $y$ is the true value based off our model which depends on the x position $x_i$ and the model paramaters ${\boldsymbol \theta}$. Finally $\mathcal{N}(0, \sigma)$ is the noise where we are makeing the assumption that our noise is normally (Gaussian) distributed.
</br>
Under the assumption of Gaussian noise we can write out the likelihood of a single data point as,
\begin{equation}
p(d_i \mid \theta) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp{\left[ \frac{-(d_i - y(x_i,{\boldsymbol \theta}))^2}{\sigma^2}\right]}
\end{equation}
</br>
The next assumption we can make is that each data point was measured independently, i.e. the measurement of one does not affect the measurement of another. When measurements are independent we can say that the joint probability is equal to the probability of each indiviual event i.e. $p(A,B,C,...) = p(A)p(B)p(C)...$. Therefore we can write the joint likelihood as,
\begin{equation}
p({\boldsymbol d} \mid {\boldsymbol \theta}) = \prod_i \frac{1}{\sqrt{2 \pi \sigma^2}} \exp{\left[ \frac{-(d_i - y(x_i,{\boldsymbol \theta}))^2}{\sigma^2}\right]}
\end{equation}
</br>
Now to find the most likeliy values, we would want to find the maximum of the probabiliy distribution $p({\boldsymbol \theta} \mid {\boldsymbol d}) \propto p({\boldsymbol \theta})p({\boldsymbol d} \mid {\boldsymbol \theta})$. 
Typically the make the computation easier we write these down as a log-prior and a log-likelihood. As the prior is flat, we can make the log-prior just return 0 as this will not affect the estimation. The log-likelihood can be written as,
\begin{equation}
\log{p({\boldsymbol d}} \mid {\boldsymbol \theta}) =  N\log{\left(\frac{1}{\sqrt{2 \pi \sigma^2}}\right)} \sum_i^N \left[ \frac{-(d_i - y(x_i,{\boldsymbol \theta}))^2}{\sigma^2}\right]
\end{equation}
This is where similarities are drawn with least squares fitting, as least squares minimises the sum in the log-likelihood. However, writing this out fully allows us with much more flexibilty and can estimate the full probability distribution of the parameters.

Now as the posterior distribution cannot be solved analytically for the majority of cases, i.e. we cannot find a mathematical solution of the posterior distribution, we have to sample the posterior distribution. This involves building up the distribution by calculating it at various points in the parameter space. The technique that we use to do this is called Markov Chain Monte Carlo (MCMC). The Monte Carlo part of this would involve calulating the posterior at random points in parameters space to build up the distribution, however, for anything but the most simple of cases this is no a feasible way to calculate the posterior. This is where the Markov Chain part solves this problem, a markov chain mean that at any point in a chain of calculations it only depends on the previous point. MCMC is a combination of these, and whilst this can be treated as a black box, some understanding may be useful. See the end of this notebook for a more in depth explanation of this process.

### Worked example

Initially we can generate our data of x and y data which is frequency and volts, where we expect a resonance peak within a backgound of data. We define the resonance peak to follow a Lorentz distribution (we wouldnt necessarily know this but assume this is true). This data could be anything, this is just an example.


```python
xdata = np.linspace(100.,101., 1000)
# Lorentz parameters
A = -0.7 # Amplitude
mu = 100.5 # mean
gam = 0.03 # width
b0 = 1
b1 = 0
sig = 0.01
# observed y data
truths = [A,mu,gam, b0,b1,sig]
ydata = A/(1 + ((xdata-mu)/gam)**2) + b0 + (xdata-mu)*b1 + np.random.normal(loc=0,scale=sig, size = (len(xdata))) 
# added Gaussian (normal) noise with 0 mean and a background of 1
```


```python
# plot the data to have a look
fig, ax = plt.subplots(figsize = (10,10))
ax.plot(xdata,ydata)
```


    
![png](20191016_Bayesian_tutorial_emcee_10_1.png)
    


The next stage is to define our model, we expect that our resonance peak follows a Lorentz distribution, the data also has some background which we define as a first order polynomial.
\begin{equation}
f(x; A, \mu, \gamma, b_0,b_1) = \frac{A \gamma}{\gamma + (x - \mu)^2} + b0 + b0(x - \mu),
\end{equation}
where $A$ is the amplitude, $\mu$ is the mean of the lorentz and $b_0, b_1$ define the background.


```python

def lorentz(x,A,mu,gam):
    """Lorentz function"""
    return A/(1 + ((x-mu)/gam)**2)

def background(x,mu,b0,b1):
    """Background function"""
    return b0 + b1*(x-mu) 

def model_lor(p0,x):
    """Entire model of data"""
    A,mu,gam,b0,b1 = p0
    mu = mu
    gam = gam
    return lorentz(x,A,mu,gam) + background(x,mu,b0,b1)
```

The next stage is to write out our Gaussian log likelihood as
\begin{equation}
\log{p({\boldsymbol d}} \mid {\boldsymbol \theta}) =  \frac{-N}{2}\log{\left(\frac{1}{\sqrt{2 \pi \sigma^2}}\right)}  + \sum_i^N \left[ \frac{-(d_i - y(x_i,{\boldsymbol \theta}))^2}{\sigma^2}\right]
\end{equation}
where our model $y(x \mid {\boldsymbol \theta})$ is the model $f(x; A, \mu, \gamma, b_0,b_1)$ above where our parameters are $A, \mu, \gamma, b_0,b_1$. We choose a Gaussian likelihood as we expect the noise to be distributed as a Gaussian, i.e. if we took a histogram of just the noise measurements with no signal this would follow a gaussian distribution. This is not always true, but for the many of cases the noise can be approximated to be Gaussian.

Whilst there are many different samplers and packages to use for MCMC we will use a package called emcee for this


```python
def log_likelihood(params,x,y,model):
    """Gaussian log likelihood"""
    modelpars = params[:-1]
    N = len(x)
    # estimate variance of noise as well
    sig = params[-1]
    
    # first term in log-Gaussian
    fact = -(N/2)*np.log(sig*sig*2*np.pi)
    
    lik_func = fact + np.sum(-((y - model(modelpars,x))**2/(2*sig*sig)))
    return lik_func
```


```python
def log_prior(params,limits):
    """uniform prior in range"""
    prior = 0
    for i,param in enumerate(params):
        #for each parameters if it its outside prior limits return 0 probability otherwise flat
        if param < limits[i][0] or param > limits[i][1]:
            return -np.inf
        else:
            # not a correctly normalised prior but is ok as an example
            prior = 1./len(params)
    return np.log(prior)
```


```python
def log_probability(params,x,y,model,limits):
    """Total log probabilty (numerator of bayes)"""
    return log_likelihood(params,x,y,model) + log_prior(params, limits)
```


```python
def emcmc_lorentz(x,y,model,limits ):
    
    # set up mcmc
    ndim, nwalkers = 6, 100
    # initial positions should be distributed across the prior space
    pos = [np.random.uniform(0,1,size=ndim)*(limits[:,1]-limits[:,0]) + limits[:,0] for i in range(nwalkers)]
    
    # set up sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x,y,model, limits))
    
    # run sampler 1000 times
    out = sampler.run_mcmc(pos, 2000)
    
    # take samples after the first 500
    samples = sampler.chain[:, 500:, :].reshape((-1, ndim))
    
    return samples
```

Once we have some guesses for the parameters we can put them through and initial maximum likelihood estimate to get better guesses of the parameters, in this case this part is similar to doing least squares. Then these parameter values can be passed to the MCMC program.


```python
# parametesr: A, mu, gam, b0, b1, sig (variance of gaussian noise)
limits = np.array([[-1,0],[100.,101.], [0,1], [0,2], [-1,1], [0,1]]) # somewhat arbritrary limits here, use appropriate ones for model
samples = emcmc_lorentz(xdata,ydata,model=model_lor, limits=limits)
#print(result)
```

The output is then samples from the posterior distribution of our parameters, this is shown in the corner plot below. Each subplot is a slice through the 5 dimensional distribution and can give us information on how correlated parameters are.


```python
parlist = ["A", "mean","gam","b0","b1","sigma"]
ig = corner.corner(samples, labels=parlist,truths=truths,truth_color="C1", color="C0",quantiles=(0.16, 0.84),levels=(1-np.exp(-0.5),1-np.exp(-2*0.5),1-np.exp(-5*0.5)))

```


    
![png](20191016_Bayesian_tutorial_emcee_22_0.png)
    


To generate the fitted curve we can estimate the true values of the parameters as the 50th percentile, i.e. 50 % of the parameter values are above this value. Then for the error bound we can set the upper as the 95% quantile and the lower as the 5% quantile. 


```python
def find_vals(xval,samples,func):
    low,mid,high = [],[],[]
    ind_list = np.array(np.random.rand(2000)*(len(samples))).astype(int)
    val_list = []
    for i in ind_list:
        #A,mu,gam,b0,b1 = samples[i]
        lor = func(samples[i],xval)
        val_list.append(lor)
    val_list = np.array(val_list).T
    for x in range(len(xval)):        
        qnt = np.percentile(val_list[x], [5,50,95])
        low.append(qnt[0])
        mid.append(qnt[1])
        high.append(qnt[2])
    
    return low,mid,high
```


```python
fits = find_vals(xdata,samples[:,:5],model_lor)
fitdat = [fits[0],fits[1],fits[2]]
```


```python
fig,ax = plt.subplots(figsize=(15,8))
ax.plot(xdata,ydata,color="C0",marker=".",ls="none")
ax.plot(xdata,fitdat[1],color = "C1",lw=3,label="Lorentz with estimated parameters")
ax.fill_between(xdata,fitdat[0],fitdat[2], color="C1", alpha=0.5)
ax.plot(xdata,model_lor(truths[:-1],xdata),"k")
ax.set_xlabel("Frequency")
ax.set_ylabel("Volts")
```




    Text(0, 0.5, 'Volts')




    
![png](20191016_Bayesian_tutorial_emcee_26_1.png)
    


This obviously fits very well so its hard to see the confidence region, i.e. the error on the fit

As an example of how to manipulate the parameters distribution, it is as simple as performin the operation on each sample individually and the errors propagate automatically. Here we calculate the value of Q which is ,
\begin{equation}
Q = \frac{\mu}{{\rm FWHM}} = \frac{\mu}{2\gamma}
\end{equation}



```python
def get_Q(samples):
    #Qfits = find_vals([0],samples,lambda p0,x: p0[1]/p0[2])
    Q_samples = [i[1]/(2*i[2]) for i in samples]
    qnt = np.percentile(Q_samples, [5,50,95])
    return Q_samples, qnt
```


```python
Q_samples, qnt = get_Q(samples)
```


```python
print(qnt[1],qnt[1]-qnt[0],qnt[2]-qnt[1])
```

    1657.9332998956083 12.434361425933503 12.460990029467212


The value of Q can then be written as,
\begin{equation}
Q = (1.77\substack{+0.03 \\ -0.03}) \cdot 10^{5}
\end{equation}


```python
fig,ax = plt.subplots(figsize=(10,5))
h = ax.hist(Q_samples,bins=50,histtype="step")
ax.axvline(qnt[0],color="C2")
ax.axvline(qnt[1],color="C1")
ax.axvline(qnt[2],color="C2")
ax.set_xlabel("Q")
ax.set_ylabel("count")
ax.get_xaxis().get_major_formatter().set_useOffset(False)
```


    
![png](20191016_Bayesian_tutorial_emcee_33_0.png)
    



```python

```
