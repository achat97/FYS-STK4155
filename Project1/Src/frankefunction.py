from functions import *


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Make data.
N = 20
noise_tuning = 0
x,y,z = Data(N,noise_tuning)

# Plot the surface.
plt.style.use('seaborn-v0_8')
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,linewidth=0, antialiased=False)

# Customize axis.
plt.setp( ax.get_xticklabels(), visible=False)
plt.setp( ax.get_yticklabels(), visible=False)
plt.setp( ax.get_zticklabels(), visible=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_zlabel(r'$z$')

plt.show()

print("Maximum z-value: ",np.max(z))
print("Minimum z-value: ",np.min(z))
