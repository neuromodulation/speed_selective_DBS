import matplotlib as mpl
import matplotlib.pyplot as plt

# Export text as regular text instead of paths or using svgfonts
mpl.rcParams['svg.fonttype'] = 'none'

# Set font to a widely-available but non-default font to show that Affinity
# ignores font-family
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Helvetica'

# Make the figure 1in x 0.75in so that incorrect font sizes in Affinity are
# very obvious (text will badly overflow).
plt.figure(figsize=(1, 0.75))

plt.plot([1, 2, 3], [2, 1, 1], label='My line')
plt.ylabel('Y-axis label')
plt.xlabel('X-axis label')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.savefig('matplotlib-demo.svg', bbox_inches='tight')