from functions import *


# Load the terrain
terrain,x,y,z = Data_terrain("SRTM_data_Morocco.tif",30,600,2600)
z_scaled = (z-np.mean(z.ravel()))
print(np.shape(x.ravel()))

plt.figure()
plt.imshow(terrain, cmap="gray")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

plt.figure()
plt.imshow(z, cmap="gray")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

plt.style.use('seaborn-v0_8')
surf = ax.plot_surface(x, y, z, cmap=cm.gist_earth,linewidth=0, antialiased=False)

# Customize axis.
plt.setp( ax.get_xticklabels(), visible=False)
plt.setp( ax.get_yticklabels(), visible=False)
plt.setp( ax.get_zticklabels(), visible=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()
