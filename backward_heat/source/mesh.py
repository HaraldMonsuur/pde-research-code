from ngsolve import Mesh
def construct_interval(N=16, T=1):
    from netgen.meshing import Element0D, Element1D
    from netgen.meshing import Mesh as NGMesh
    from netgen.meshing import MeshPoint, Pnt
    ngmesh = NGMesh(dim=1)
    pids = []
    for i in range(N + 1):
        pids.append(ngmesh.Add(MeshPoint(Pnt(T * i / N, 0, 0))))
    for i in range(N):
        ngmesh.Add(Element1D([pids[i], pids[i + 1]], index=1))
    ngmesh.Add(Element0D(pids[0], index=1))
    ngmesh.Add(Element0D(pids[N], index=2))
    ngmesh.SetBCName(0, "start")
    ngmesh.SetBCName(1, "end")
    return ngsolve.Mesh(ngmesh)
from netgen.meshing import *
from netgen.csg import *
import ngsolve

def MakeStructured2DMesh(quads=True, nx=10, ny=10, secondorder=False, periodic_x=False, periodic_y=False, mapping = None, bbpts=None, bbnames=None, flip_triangles=False, boundarylayer=None, hppnts=None):
    """
    Generate a structured 2D mesh

    Parameters
    ----------
    quads : bool
      If True, a quadrilateral mesh is generated. If False, the quads are split to triangles.

    nx : int
      Number of cells in x-direction.

    ny : int
      Number of cells in y-direction.

    secondorder : bool
      If True, second order curved elements are used.

    periodic_x: bool
      If True, the left and right boundaries are identified to generate a periodic mesh in x-direction.

    periodic_y: bool
      If True, the top and bottom boundaries are identified to generate a periodic mesh in y-direction.

    mapping: lamda
      Mapping to transform the generated points. If None, the identity mapping is used.
    
    bbpts : list
      List of points which should be handled as BBND and are named with bbnames. The mesh (nx, ny and mapping) must be constructed in such a way that the bbpts coincide with generated points. Otherwise an Exception is thrown.

    bbnames : list
      List of bbnd names as strings. Size must coincide with size of bbpts. Otherwise an Exception is thrown.

    flip_triangles : bool
      If set to True together with quads=False the quads are cut the other way round

    boundarylayer : dict
      If not None it expects a dictionary of the form { "boundaryname" : [t1,...,tn] } where ti denote the thickness of layer i. The number of layers are included in nx/ny. After the layers are placed the remaining number of cells are used to divide the remaining grid uniformly.

    hppnts : list
      If not None it expects a list of the form [ (px1,py1, hpref1), (px2,py2, hpref2), ... ] where px,py are the point coordinates which have to be resolved in the mesh and hpref the refinement factor

    Returns
    -------
    (ngsolve.mesh)
      Returns generated 2D NGSolve mesh

    """
    mesh = Mesh()
    mesh.dim=2

    if (bbpts and bbnames) and len(bbpts) != len(bbnames):
        raise Exception("Lenght of bbnames does not coincide with length of bbpts!")

    found = []
    indbbpts = []
    if bbpts:
        for i in range(len(bbpts)):
            found.append(False)
            indbbpts.append(None)
    foundhp = [ False for i in hppnts] if hppnts else []
        

    pids = []
    if periodic_y:
        minioni = []
        masteri = []
    if periodic_x:        
        minionj = []
        masterj = []
        
    numlayerleft  = len(boundarylayer["left"]) if (boundarylayer and boundarylayer.get("left")) else 0
    numlayerright = len(boundarylayer["right"]) if (boundarylayer and boundarylayer.get("right")) else 0
    numlayertop   = len(boundarylayer["top"]) if (boundarylayer and boundarylayer.get("top")) else 0
    numlayerbot   = len(boundarylayer["bottom"]) if (boundarylayer and boundarylayer.get("bottom")) else 0

    thicknessleft  = [0]
    thicknessright = [0]
    thicknesstop   = [0]
    thicknessbot   = [0]
    for i in range(numlayerleft):
        thicknessleft.append(thicknessleft[-1]+boundarylayer["left"][i])
    for i in range(numlayerright):
        thicknessright.append(thicknessright[-1]+boundarylayer["right"][i])
    for i in range(numlayertop):
        thicknesstop.append(thicknesstop[-1]+boundarylayer["top"][i])
    for i in range(numlayerbot):
        thicknessbot.append(thicknessbot[-1]+boundarylayer["bottom"][i])

        
    for i in range(ny+1):
        for j in range(nx+1):
            x = thicknessleft[j] if j < numlayerleft else ((thicknessleft[-1]+(j-numlayerleft)/(nx-numlayerleft-numlayerright)*(1-thicknessleft[-1]-thicknessright[-1])) if j < nx-numlayerright else 1-thicknessright[nx-j])
            y = thicknessbot[i] if i < numlayerbot else ((thicknessbot[-1]+(i-numlayerbot)/(ny-numlayerbot-numlayertop)*(1-thicknessbot[-1]-thicknesstop[-1])) if i< ny-numlayertop else 1-thicknesstop[ny-i])
            pids.append(mesh.Add (MeshPoint(Pnt(x,y,0))))
            if periodic_y:
                if i == 0:
                    minioni.append(pids[-1])
                if i == ny:
                    masteri.append(pids[-1])  
            if periodic_x:                       
                if j == 0:
                    minionj.append(pids[-1])
                if j == nx:
                    masterj.append(pids[-1])        
    if periodic_y:
        for i in range(len(minioni)):   
            mesh.AddPointIdentification(masteri[i],minioni[i],identnr=1,type=2)
    if periodic_x:            
        for j in range(len(minionj)):        
            mesh.AddPointIdentification(masterj[j],minionj[j],identnr=2,type=2)                                       

    # mesh.Add(FaceDescriptor(surfnr=1,domin=1,bc=1))
    idx_dom = mesh.AddRegion("dom", dim=2)
    idx_bottom = mesh.AddRegion("bottom", dim=1)
    idx_right  = mesh.AddRegion("right", dim=1)
    idx_top    = mesh.AddRegion("top", dim=1)
    idx_left   = mesh.AddRegion("left", dim=1)
    
    for i in range(ny):
        for j in range(nx):
            base = i * (nx+1) + j
            if quads:
                pnum = [base,base+1,base+nx+2,base+nx+1]
                elpids = [pids[p] for p in pnum]
                el = Element2D(idx_dom,elpids)
                if not mapping:
                    el.curved=False
                mesh.Add(el)
            else:
                if flip_triangles:
                    pnum1 = [base,base+1,base+nx+2]
                    pnum2 = [base,base+nx+2,base+nx+1]
                else:
                    pnum1 = [base,base+1,base+nx+1]
                    pnum2 = [base+1,base+nx+2,base+nx+1]
                elpids1 = [pids[p] for p in pnum1]
                elpids2 = [pids[p] for p in pnum2]
                mesh.Add(Element2D(idx_dom,elpids1)) 
                mesh.Add(Element2D(idx_dom,elpids2))                          

    for i in range(nx):
        mesh.Add(Element1D([pids[i], pids[i+1]], index=idx_bottom))
    for i in range(ny):
        mesh.Add(Element1D([pids[i*(nx+1)+nx], pids[(i+1)*(nx+1)+nx]], index=idx_right))
    for i in range(nx):
        mesh.Add(Element1D([pids[ny*(nx+1)+i+1], pids[ny*(nx+1)+i]], index=idx_top))
    for i in range(ny):
        mesh.Add(Element1D([pids[(i+1)*(nx+1)], pids[i*(nx+1)]], index=idx_left))

    # mesh.SetBCName(0, "bottom")        
    # mesh.SetBCName(1, "right")        
    # mesh.SetBCName(2, "top")        
    # mesh.SetBCName(3, "left")  

    mesh.Compress()       
    
    if secondorder:
        mesh.SecondOrder()
    
    if mapping:
        for p in mesh.Points():
            x,y,z = p.p
            x,y = mapping(x,y)
            p[0] = x
            p[1] = y

    for k in range(len(found)):
        i = 0
        for p in mesh.Points():
            if abs(p.p[0]-bbpts[k][0])+abs(p.p[1]-bbpts[k][1]) < 1e-6:
                indbbpts[k] = pids[i]
                found[k] = True
            i += 1
    for k in range(len(found)):
        if found[k] == False:
            raise Exception("bbpnt[",k,"] not in structured mesh!")
    for i in range(len(indbbpts)):
        mesh.Add(Element0D(indbbpts[i], index=i+1))
        mesh.SetCD2Name(i+1, bbnames[i])

    for k in range(len(foundhp)):
        i = 0
        for p in mesh.Points():
            if abs(p.p[0]-hppnts[k][0])+abs(p.p[1]-hppnts[k][1]) < 1e-6:
                mesh.AddSingularity(pids[i],hppnts[k][-1])
                foundhp[k] = True
            i += 1
    for k in range(len(foundhp)):
        if foundhp[k] == False:
            raise Exception("hppnts[",k,"] not in structured mesh!")
    
            
    ngsmesh = ngsolve.Mesh(mesh)
    return ngsmesh



def construct_2d_square_mesh(nrefines=1):
    from ngsolve import unit_square
    mesh = ngsolve.Mesh(unit_square.GenerateMesh(maxh=1))
    mesh.Refine()
    #mesh = MakeStructured2DMesh(quads=False, nx=2, ny=2)
    for _ in range(nrefines):
        mesh.Refine()
    return mesh, "bottom|right|top|left"


def construct_3d_cube_mesh(nrefines=1):
    from netgen.csg import CSGeometry, OrthoBrick, Pnt
    cube = OrthoBrick(Pnt(0, 0, 0), Pnt(1, 1, 1))
    geo = CSGeometry()
    geo.Add(cube, maxh=1)
    ngmesh = geo.GenerateMesh()
    ngmesh.Refine()
    mesh = Mesh(ngmesh)
    for _ in range(nrefines):
        mesh.Refine()
    return mesh, "default"
