import msapy.firstorderelastic3D as mp

coord, connectivity, fixity, \
nodal_load, member_load, section = mp.excel_preprocessor('input.xlsx')

structure1 = mp.Structure_3d1el(coord,fixity,connectivity,
                               nodal_load,member_load,section)
structure1.solve()

fig = structure1.plot()
fig.show()