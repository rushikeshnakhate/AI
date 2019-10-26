https://brilliant.org/wiki/bayes-theorem/


P(H∣E)= P(E∣H)/P(E)  *	P(H).

prior probability
This relates the probability of the hypothesis before getting the evidence P(H), to the probability of the hypothesis after getting the evidence, P(H∣E). For this reason, P(H) is called the prior probability

posterior probability
 P(H∣E) is called the posterior probability
 
likelihood ratio
The factor that relates the two, P(E∣H)/p(E)is called the likelihood ratio. 

Using these terms, Bayes' theorem can be rephrased as "the posterior probability equals the prior probability times the likelihood ratio."







few important points 
posterior = likelihood *  prior /norm_marginal 
P(E---> is given so would come in denom always ) 







Introduction and Review.
Bayesian Regression comes with a different toolset than ordinary Linear Regression. In turn, that toolset demands a slightly different mindset. We start with a short review to hightlight the ways in which Bayesian thinking proceeds.

Consider a population whose age distribution is as follows:

Age group	       %%  of total population
≤35≤35 	         25%25% 
36−6536−65 	     45%45% 
≥66≥66 	         30%30% 

Say you know the following results of a study about YouTube viewing habits:

Age group	          %%  in this group that watch YouTube every day
≤35≤35             	 90%90% 
36−6536−65         	 50%50% 
≥66≥66             	 10%10% 
Prompt: If you know a user watches YouTube every day, what is the probability that they are under 35?

We will start with a prior, then update that prior using the likelihood and the normalization from Bayes's formula. We define the following notation:

AA : YouTube watching habit
BB : Age
A=1A=1 : User watches YouTube every day
A=0A=0 : User does not watch YouTube every day
B≤35B≤35 : User has age between 0 and 35
36≤B≤6536≤B≤65 : User has age between 36 and 65
B≥66B≥66 : User has age greater than 65
The prior can be read from the first table:

P(B≤35)=0.25
P(B≤35)=0.25
 
We are looking to calculate the posterior probability:

P(B≤35|A=1)
P(B≤35|A=1)
 
With Bayes's formula:

P(B|A)=P(A|B)P(B)P(A)
P(B|A)=P(A|B)P(B)P(A)
 
For our question:

P(B≤35|A=1)=P(A=1|B≤35)∗P(B≤35)P(A=1)
P(B≤35|A=1)=P(A=1|B≤35)∗P(B≤35)P(A=1)
 
While the tables do not contain the value of  P(A=1)P(A=1)  it may be calculated:

P(A=1)=P(A=1)= P(A=1|B≤35)∗P(B≤35) +            P(A=1|B≤35)∗P(B≤35) + 
               P(A=1|35<B<65)∗P(35<B<65) +            P(A=1|35<B<65)∗P(35<B<65) + 
               P(A=1|B≥65)∗P(B≥65)            P(A=1|B≥65)∗P(B≥65)
            
            





In real life, this situation corresponds to:

Surveying people and asking them two questions (what's your age? do you watch YouTube every day?), then tabulating the percentage of each age group that watch YouTube everyday.
After having collected that data, observing the anonymized watching habits of a set of different users (not the survey takers) - without access to additional demographic info - and using the above survey to derive a probability for the anonymized users' age.



Bayesian Linear Regression
In Bayesian Linear Regression, our prior expresses a belief about the parameters of linear regression we wish to calculate, 
1. the linear coefficient vector should have a small absolute value, 
2.  deviations from zero should be Gaussian. 
This prior is mathematically equivalent to the Ridge Regression condition.





So the question we will now ask is: conditioned on the data, how does our belief regarding the parameters of linear regression change?
This is the same prior-to-posterior calculation of the above exercise.
Bayes' Rule:

P(B|A)=P(A|B) P(B)P(A)P(B|A)=P(A|B) P(B)P(A) ,

For linear Regression:
p(w|y,X)=p(y|w,X) p(w)p(y|X)p(w|y,X)=p(y|w,X) p(w)p(y|X) 
What do we know, and what do we not know?


p(w)=N(0,λ−1I)p(w)=N(0,λ−1I) : That's the prior on  ww  --- Known.
p(y|w,X)=N(Xw,σ2I)p(y|w,X)=N(Xw,σ2I) : That's the likelihood expression --- Known.
p(y|X)p(y|X) : That's the marginal probability of  yy  --- NOT KNOWN
Rewriting the marginal probability in detail, using an integral instead of a sum - since  ww  is a continuous variable.

p(y|X)=∫ℝ𝕕p(y,w|X) dwp(y|X)=∫Rdp(y,w|X) dw 
          =∫ℝ𝕕 p(y|w,X) p(w) dw          =∫Rd p(y|w,X) p(w) dw 
At this point approximation is frequently required as the above integral usually has no closed form.





Coding Bayesian Linear Regression , an equation for the posterior probability of  ww , the linear regression parameter vector:
p(w|y,X)=N(w|μ,Σ)
p(w|y,X)=N(w|μ,Σ)
 
where
Σ=(λ I+σ−2 XT X)−1
Σ=(λ I+σ−2 XT X)−1
 
μ=(λ σ2I+XT X)−1 XTy⇐wMAP
μ=(λ σ2I+XT X)−1 XTy⇐wMAP
 
Recall that  σ2σ2  is a parameter characterizing the deviation of the data from the line defined by  XwXw . While we don't know the true underlying parameter, we can estimate it by using the empirical deviation:

σ2≈σ̂ 2=1n−dΣni=1(yi−Xiw)2
σ2≈σ^2=1n−dΣi=1n(yi−Xiw)2
 

Where  ww  in the above is the  wLeastSquares=(XT X)−1 XTywLeastSquares=(XT X)−1 XTy 
When it comes to prediction:
p(y0|x0,y,X)=N(y0|μ0,σ20)p(y0|x0,y,X)=N(y0|μ0,σ02) 
                μ0=xT0μ                μ0=x0Tμ 
                σ20=σ2+xT0Σx0                σ02=σ2+x0TΣx0 

