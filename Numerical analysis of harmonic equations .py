#!/usr/bin/env python
# coding: utf-8

# In[13]:


#importing necessary packages
from sympy.interactive import printing 
printing.init_printing(use_latex=True)
from sympy import * 
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


# # Q1  **(a)**

# Here we find the general solution for for second order linear differencial equation that is homogeneous with constant coefficents 
# <br>
# <br>
# in class we solved y'' +y=0 and we realized that y can take two forms either exponetial 'e^kx'or sin 
# <br>
# <br>
# This is because when we plag either to the equation and solve then we end up with 0 as expected
# <br>
# <br>
# so we expect our solution-general solution- to be in the form of:
# <br>
# <br>
# y=Asin()+Bsin() where A and B are coeffcients or 
# <br>
# <br>
# y=Ae^ωk+Be^ωk
# <br>
# <br>
# Either of the above forms is correct

# In[14]:


#espressing ω,t as symbols that we will use later where omega where ω=sqrt(k/m)
ω,t=sp.symbols(' ω t')
#expressing x as a fuction
x=sp.Function('x')


# In[15]:


#here our fuction is dependant on t 
diff_2=Eq(sp.Derivative(x(t), t, 2) + ω**2 * x(t),0)
#displaying our fuction
display(diff_2)


# The General solution is as below where **C1** and **C2**  are coeffcients 

# In[16]:


# here we dislove the fuction which means we differenciate with respect to t , solving the equation gives us the general solution
#C1 and C2 on the display are place holders 
ans= sp.dsolve(sp.Derivative(x(t), t, 2) + ω**2 * x(t))
print("\n General solution\n" )
display(ans)


# **-------------------------------------------------------------------------------------------------------------------**

# We solve for our two unkowns c1 and c2
# <br>
# We also substitute relevant variables with intial condition values where
# <br>
# ω0 = 2,
# <br>
# x(0) =1, 
# <br>
# x'(0) = 0

# In[17]:


initial_con = [sp.Eq(ans.args[1].subs(t, 0), 1), sp.Eq(ans.args[1].diff(t).subs(t, 0), 0)]
initial_con


# In[18]:


initial_con_ans_1=sp.solve(initial_con)
initial_con_ans_1


# The below equation is as a result of solving for the cieffcients and substitutin with there values: 
# <br>
# <br>
# we then simplified the equation to obtain a simpler version of x(t)
# <br>
# <br>
# The complex exponentials are not a surprise as we expected an oscillatory fuction from the natural behaviour of springs # **statistical Mechanic NS162**

# In[19]:


initial_con_full_ans = ans.subs(initial_con_ans_1[0])
initial_con_full_ans 


# We substitute the values
# <br>
# <br>
# ω0 = 2,
# <br>
# <br>
# x(0) =1,
# <br>
# <br>
# x'(0) = 0
# <br>
# <br>
# Given in the problem and simplify to obtaing the below fuction

# In[20]:


ẋ, x_0=sp.symbols(' ẋ  x_0')
given_scenario = sp.simplify(initial_con_full_ans.subs({x_0:1, ẋ:0, ω:2}))
given_scenario 


# We then plot the simplified **x(t)** over range 

# In[21]:


sp.plot(given_scenario.rhs)


# The plot above follows the naturla behaviour of harmonic oscillators 
# <br>
# <br>
# **--------------------------------------------------------------------------------------------------------------------------------------**

# # Q1 **(b)**

# Here we implement ous solution using the **complexification technique**
# <br>
# <br>
# That is :
# u=(x+yi)^n where yi is the imaginary part and x is the real part

# In[22]:


ω_1=sp.symbols('ω_1')


# In[23]:


diff_1=Eq(sp.Derivative(x(t), t, 2) + ω**2 * x(t),cos(ω_1*t))
display(diff_1)


# We already know the genral solution from part a here we just need to include the real part 

# In[24]:


complification=dsolve(Eq(sp.Derivative(x(t), t, 2) + ω**2 * x(t),cos(ω_1*t)))
print("\n General solution\n" )
display(complification)


# We solve for our two unkowns c1 and c2
# <br>
# We also substitute relevant variables with intial condition values where
# <br>
# ω0 = 2,
# <br>
# x(0) =1, 
# <br>
# x'(0) = 0

# In[25]:


complification_sol= [sp.Eq(complification.args[1].subs(t, 0), 1), sp.Eq(complification.args[1].diff(t).subs(t, 0), 0)]
display(complification_sol)


# We substitute the values
# <br>
# <br>
# ω0 = 2,
# <br>
# <br>
# x(0) =1,
# <br>
# <br>
# x'(0) = 0
# <br>
# <br>
# Given in the problem and simplify to obtaing the below fuction

# In[26]:


complification_sol_1=sp.solve(complification_sol)
display(complification_sol_1)


# The below equation is as a result of solving for the cieffcients and substitutin with there values: 
# <br>
# <br>
# we then simplified the equation to obtain a simpler version of x(t)
# <br>
# <br>
# The complex exponentials are not a surprise as we expected an oscillatory fuction from the natural behaviour of springs # **statistical Mechanic NS162**

# In[27]:


complification_sol_full = ans.subs(complification_sol_1[0])
display(complification_sol_full )


# ω1=! ω0 because the fuction will be undefined, the denominator will be 0 and this we can not solve the problem

# **--------------------------------------------------------------------------------------------------------**
# <br>
# <br>
# We substitute the values
# <br>
# <br>
# ω0 = 2,
# <br>
# <br>
# x(0) =1,
# <br>
# <br>
# x'(0) = 0
# <br>
# <br>
# Given in the problem and simplify to obtaing the below fuction

# In[28]:


given_scenario_2 = sp.simplify(complification_sol_full.subs({x_0:1, ẋ:0, ω:2,ω_1:3}))
given_scenario_2 


# # Q1 **(c)**

# We then plot the simplified **x(t)** over range 

# In[29]:


sp.plot(given_scenario_2.rhs)


# The plot above follows the naturla behaviour of harmonic oscillators 
# <br>
# <br>
# **--------------------------------------------------------------------------------------------------------------------------------------**

# # We repeat Q1 b and c using sin instead, all fuctions are the same just replace cos with sin, and the expanantions are same too. 

# In[30]:


diff_1_sin=Eq(sp.Derivative(x(t), t, 2) + ω**2 * x(t),sin(ω_1*t))
display(diff_1_sin)


# In[31]:


complification_sin=dsolve(Eq(sp.Derivative(x(t), t, 2) + ω**2 * x(t),sin(ω_1*t)))
display(complification_sin)


# In[32]:


complification_sol_sin= [sp.Eq(complification_sin.args[1].subs(t, 0), 1), sp.Eq(complification_sin.args[1].diff(t).subs(t, 0), 0)]
display(complification_sol_sin)


# In[33]:


complification_sol_1_sin=sp.solve(complification_sol_sin)
display(complification_sol_1_sin)


# In[34]:


complification_sol_full_sin = ans.subs(complification_sol_1_sin[0])
display(complification_sol_full_sin )


# In[35]:


given_scenario_2_sin = sp.simplify(complification_sol_full_sin.subs({x_0:1, ẋ:0, ω:2,ω_1:3}))
given_scenario_2_sin 


# In[36]:


sin_plot=sp.plot(given_scenario_2_sin.rhs)
sin_plot


# # Q1 **(d)**

# we set x=f(t)e^(iωt): we do this because solving it directly leads to undefined solution as we get 0 at the denominator

# x=f(t)e^(iωt)
# <br>
# <br>
# x'=f'(t)e^(iωt)+iωtf'(t)e^(iωt)
# <br>
# <br>
# x''=f''(t)e^(iωt)+iωtf'(t)e^(iωt)+iωtf'(t)e^(iωt)-ω^2 * f(t)e^(iωt)
# <br>
# <br>
# x''+ω^2*x=f"(t)e^(iωt)+2iωf'(t)e^(iωt)-ω^2 * f(t)e^(iωt)+ω^2 * f(t)e^(iωt)
# <br>
# <br>
# The last terms disapear
# <br>
# <br>
# f"(t)e^(iωt)+2iω* f(t)e^(iωt)
# <br>
# <br>
# dislvoing the fuction further yields
# e^iω
# <br>
# <br>
# e^iω=cosω+isinω
# <br>
# <br>
# general solution 
# <br>
# <br>
# e^iω or cosω+isinω

# In[ ]:


import cmath
import math
from math import e
i =1j


# In[ ]:


#set x=f(t)e^(iωt)
x=sp.Function('f')(t)*e**(i*ω*t)
display(x)


# In[39]:


fuc=Eq(sp.Derivative(x, t, 2) + ω**2 * x,sin(ω*t))
display(fuc)


# In[40]:


complification_ii=dsolve(Eq(sp.Derivative(x, t, 2) + ω**2 * x,sin(ω*t)))
display(complification_ii)


# # Q1 **(e)**

# The general solution differ in the coefficients as well as the placement of the imaginary part
# <br>
# <br>
# in the first one we had general solution in the form Ae^ωk+Be^ωk while in the d part we have cosω+isinω:
# 
# 

# # Q1 **(f)** Vedio explanation based on Numeroical analysis knowledge

# We know that ω=sqrt(k/m) where k is the spring constant and m is the mass. If k is big it means the spring is strong that the oscillation will be small but faster , while if m is big the ossicilation would be slower due to change of intertia. Applying the same knowledge on the brige from the 0.21 second the oscillation are slow but big which means the mass is greater than what the spring can hold thus may tend towards breaking point . Some sections of the brige are vibrating faster but the amplitude is smaller which means the bridge has kess weight there that is less than the spring constant. The wind acts as an external agents that acts as a catalyst, here it increased m thus the bridge reached its breaking point. Notice that it breaks fast where the vibration were slow but big (m>>k).. This show how omega which is a fuction of sqrt(k/m) contributes in the harmonic oscilation

# # Question 2

# I recorded two audio from different location 
# <br>
# <br>
# and used https://www.online-convert.com/result#j=5eaeec88-dddd-426d-a83b-4012d440a97c
#    <br>
#    <br>
# to convert the audi into wave. I then read it throw wave.open from scipy

# In[50]:


from scipy.io import wavfile
from scipy import * 
import wave
import matplotlib.pyplot as plt


# In[51]:


noise = wave.open("noise1.wav",'r')
noise_rate=np.inf


# In[52]:


sig = np.frombuffer(noise.readframes(noise_rate), dtype=np.int16)
plt.plot(sig)
plt.title("one period from noise")
plt.xlabel("sample")
plt.ylabel("amplitude")


# In[53]:


trans_1 = fft(sig)
plt.figure(figsize=(10, 6), dpi=330)
plt.plot(abs(trans_1)[:20000])
plt.ylabel("amplitude")
plt.xlabel("Hz")
plt.title("frequency decomposition with discrete fourier transform for noise 1")
plt.show()


# In[72]:


peak_1=2500+2500/2
peak_1


# The dominant peak is at around 3750.0HZ
# <br>
# <br>
# This is the highest frequency after we decompose the wave into finite number of sinusoidal, From the graph we other smaller peaks that reduce over time and flattens towards the 20000HZ

# In[54]:


noise_2 = wave.open("noise2.wav",'r')
noise_rate=np.inf


# In[55]:


sig_2 = np.frombuffer(noise_2.readframes(noise_rate), dtype=np.int16)
plt.plot(sig_2)
plt.title("one period from noise")
plt.xlabel("sample")
plt.ylabel("amplitude")


# In[73]:


from scipy.signal import find_peaks
trans_2 = fft(sig_2)
plt.figure(figsize=(10, 6), dpi=330)
peaks, _=  find_peaks(trans_2, height=0)
plt.plot(abs(trans_2)[:40000])
plt.ylabel("amplitude")
plt.xlabel("Hz")
plt.title("frequency decomposition with discrete fourier transform for noise 2")
plt.show()


# In[71]:


peak=7500+2500/2
peak


# The dominat frequncy is at around 8750.0HZ
# <br>
# From there the frequency hight reduces over time and decreases more between 35000 and 40000. This means that if I was studying and got rid of 8750.0 then noise would reducee significantly . It is at the dorminat frequency where the highest energy is found E = hν = hc/λ, where E = energy, h = Planck's constant, ν = frequency, c = the speed of light, and λ = wavelength. # **statistical Mechanic NS162** . Removing the highest frequency means that noise measured in decibels would be lower

# From the second gragh I would need earbud that cancels frequncy btn 2500Hz to 35000Hz as its the region with more fluctuation and high average enegy thus noise

# From the 1st gragh the noise cancellation should be able to cancel frequencies between 2500Hz to 17500Hz

# In[ ]:




