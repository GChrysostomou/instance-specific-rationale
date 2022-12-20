from ast import comprehension
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FA_name = "attention"
df = pd.read_csv(f'sst_{FA_name}.csv')
comp = df["F-Comp"]
soft = df["F-SoftComp"]
SET=df.index
# Initialize figure and axis
fig, ax = plt.subplots(figsize=(5, 5))

# Plot lines
ax.plot(SET, comp, color="red")
ax.plot(SET, soft, color="green")

# Fill area when income > expenses with green
ax.fill_between(
    SET, comp, soft, where=(soft > comp), 
    interpolate=True, color="green", alpha=0.25, 
    label="Positive"
)

# Fill area when income <= expenses with red
ax.fill_between(
    SET, comp, soft, where=(soft <= comp), 
    interpolate=True, color="red", alpha=0.25,
    label="Negative"
)




ax.set_xlabel('Replaced tokens')
ax.set_ylabel('f(i) = |M(So)-M(Si)| / |M(So)-M(S6)|')
ax.set_title('Interpolation Analysis')








ax.legend()
plt.show()
