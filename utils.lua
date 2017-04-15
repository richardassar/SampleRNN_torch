function linear2mu(x,mu) -- [-1,1] -> [0,mu]
    mu = mu or 255
    return torch.floor((torch.cmul(torch.sign(x),torch.log(1+mu*torch.abs(x))/math.log(1+mu))+1)/2*mu)  
end

function mu2linear(x, mu) -- [0,mu] -> [-1,1]
    mu = mu or 255
    local y = 2*(x-(mu+1)/2)/(mu+1)
    return torch.cmul(torch.sign(y),(1/mu)*(torch.pow(1+mu,torch.abs(y))-1))
end