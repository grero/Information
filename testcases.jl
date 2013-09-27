include("information.jl")

function testcase2(ntrials::Int64, rho::Float64)
    #Two variables perfectly anti-correlated for one set of stimuli, perfectly correlated for another
    counts = rand(ntrials,2)
    counts = counts .< 0.5
    h = div(ntrials,2)
    counts[1:h,2] = (~counts[1:h,1])&(rand(h).<rho)
    counts[h+1:end,2] = counts[h+1:end,1]&(rand(h) .<rho)
    counts = int(counts)
    trial_labels = int(cat(1,ones(h),2*ones(h)))
    #independent
    E1 = computeEntropy(counts[:,1])
    E2 = computeEntropy(counts[:,2])
    Ec1 = computeConditionalEntropy(counts[:,1],trial_labels,0)
    Ec2 = computeConditionalEntropy(counts[:,2],trial_labels,0)

    #joint
    E = computeEntropy(counts)
    Ec = computeConditionalEntropy(counts,trial_labels,0)

    return E1,Ec1,E2,Ec2, E,Ec
end

testcase2(ntrials) = testcase2(ntrials,0.9)

function testcase3(ntrials::Array{Int64,1})
    n = length(ntrials)
    E1 = zeros(n)
    Ec1 = zeros(n)
    E2 = zeros(n)
    Ec2 = zeros(n)
    E = zeros(n)
    Ec = zeros(n)
    for i=1:n
        E1[i],Ec1[i],E2[i],Ec2[i],E[i],Ec[i] = testcase3(ntrials[i])
    end
    return E1,Ec1,E2,Ec2,E,Ec
end

function testcase2(ntrials::Array{Int64,1})
    n = length(ntrials)
    E1 = zeros(n)
    Ec1 = zeros(n)
    E2 = zeros(n)
    Ec2 = zeros(n)
    E = zeros(n)
    Ec = zeros(n)
    for i=1:n
        E1[i],Ec1[i],E2[i],Ec2[i],E[i],Ec[i] = testcase2(ntrials[i])
    end
    return E1,Ec1,E2,Ec2,E,Ec
end

function testcase3(ntrials::Int64)
    #Two variables with information encoded only in their mean firing rates
    #Indepdendent information
    counts = zeros(Int64,ntrials,2)
    h = div(ntrials,2)
    counts[1:h,1] = int(rand(h) .< 0.3)
    counts[h+1:end,1] = int(rand(h) .< 0.8)
    counts[1:h,2] = int(rand(h) .< 0.4)
    counts[h+1:end,2] = int(rand(h) .< 0.6)
    trial_labels = int(cat(1,ones(h),2*ones(h)))
    #independent
    E1 = computeEntropy(counts[:,1])
    E2 = computeEntropy(counts[:,2])
    Ec1 = computeConditionalEntropy(counts[:,1],trial_labels,0)
    Ec2 = computeConditionalEntropy(counts[:,2],trial_labels,0)

    #joint
    E = computeEntropy(counts)
    Ec = computeConditionalEntropy(counts,trial_labels,0)

    return E1,Ec1,E2,Ec2, E,Ec

end

function simulate(ntrials::Int64, nvars::Int64)
    #Simulate a series of independent bernoulli processes with two different 'stimulus' categories,
    #corresponding to two different p. The point of this is just to illustrate the effect of shuffling 
    #the trial labels, i.e. destroying the information about stimulus.
    #Input:
    #   ntrials     :       number of simulated trials
    #   nbins       :       number of temporal bins
    #   word_size   :       the number of bins to cancatenate when forming words
    #Output:
    #   E           :       the total entropy for each bin
    #   Ec          :       the conditional entropy for each bin
    #   Es          :       the total entropy when shuffling categories labels
    #   Ecs         :       the conditional entropy when conditioning on the wrong categories
    counts = rand(ntrials,nvars)
    counts[1:div(ntrials,2),:] = int(counts[1:div(ntrials,2),:] .< 0.2)
    counts[div(ntrials,2):end,:] = int(counts[div(ntrials,2):end,:] .< 0.6)
    counts = int(counts)
    trial_labels = int(cat(1,ones(div(ntrials,2)),2*ones(div(ntrials,2))))
    ps = [0.5;0.5]
    #compute information
    E = computeEntropy(counts)
    Ec = computeConditionalEntropy(counts,trial_labels,0)
    #compute shuffle
    Ecs = computeConditionalEntropy(counts,trial_labels,100)
    #compute the true entropies as well
    #compute the unconditional probability
    pc = [0.8 0.2; 0.4 0.6]
    p = pc'*ps
    Etrue = -p'*log2(p)
    Ectrue = -ps'*sum(pc.*log2(pc),2)
    
    return E,Ec, E, Ecs, Etrue, Ectrue
end
