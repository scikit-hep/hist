from coffea import processor
import ROOT
import math

def convert(hist,hname=""):
    """Convert a 1-dimensional or 2-dimensional`Hist` object or a counter to ROOT.TH1D or ROOT.TH2D
    Parameters
    ----------
        hist : Hist
            A 1-dimensional or 2-dimensional histogram object or a counter
        hname: new name for the output histogram
    Returns
    -------
        out
            A ``ROOT.TH1D`` or ``ROOT.TH2D`` object
    Examples
    --------
    Saving histograms in the coffea result into a ROOT file

        result = processor.run_uproot_job(
            samples,
            "Events",
            Analysis(),
            processor.iterative_executor,
            {"schema": NanoAODSchema},
        )

        import ROOT
        import hist2root

        fOUT = ROOT.TFile.Open('output.root','RECREATE')
        for k in result.keys():
            hist2root.convert(result[k],k).Write()
        fOUT.Close()
    """
    if isinstance(hist,processor.defaultdict_accumulator):
        return convertCounter(hist,hname)
        
    if len(hist.axes()) == 1:
        return convert2TH1D(hist,hname)

    if len(hist.axes()) == 2:
        return convert2TH2D(hist,hname)

def convertCounter(counter,hname):
    n = len(counter.keys())
    out = ROOT.TH1D(hname,"",n,0,n)
    for i,key in enumerate(counter.keys()):
        out.SetBinContent(i+1,counter[key])
        out.GetXaxis().SetBinLabel(i+1,key)
    return out
    
def convert2TH1D(hist,hname):
    name = hist.label
    if len(name) > 0:
        name  = hname
    title = hist.label
    axis  = hist.axes()[0]
    edges = axis.edges(overflow="none")
    edgesOFlow = axis.edges(overflow="all")

    out = ROOT.TH1D(name,title,len(edges)-1,edges)
    for i in range(0,len(edgesOFlow)-1):
        bincenter = edgesOFlow[i] + 0.5*(edgesOFlow[i+1]-edgesOFlow[i])
        x = bincenter
        w,wsq = hist.values(sumw2=True,overflow="all")[()]
        out.Fill(x,w[i])
        out.SetBinError(i,math.sqrt(wsq[i]))
    return out

def convert2TH2D(hist,hname):
    name = hist.label
    if len(name) > 0:
        name  = hname
    title = hist.label
    axis1 = hist.axes()[0]
    xedges = axis1.edges(overflow="none")
    xedgesOFlow = axis1.edges(overflow="all")
    axis2 = hist.axes()[1]
    yedges = axis2.edges(overflow="none")
    yedgesOFlow = axis2.edges(overflow="all")
    
    out = ROOT.TH2D(name,title,len(xedges)-1,xedges,len(yedges)-1,yedges)

    for i in range(0,len(xedgesOFlow)-1):
        xbincenter = xedgesOFlow[i] + 0.5*(xedgesOFlow[i+1]-xedgesOFlow[i])
        x = xbincenter
        for j in range(0,len(yedgesOFlow)-1):
            ybincenter = yedgesOFlow[j] + 0.5*(yedgesOFlow[j+1]-yedgesOFlow[j])
            y = ybincenter
            w,wsq = hist.values(sumw2=True,overflow="all")[()]
            out.Fill(x,y,w[j][i])
            out.SetBinError(i,j,math.sqrt(wsq[j][i]))
    return out
