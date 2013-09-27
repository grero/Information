module Information

export computeJointInformation,computeEntropy,computeJointEntropy, computeTemporalEntropy

using MAT

function saveInformationResults(fname::String, counts::Array{Int64,3}, H::Array{Float64,1}, Hc::Array{Float64,1}, Hi::Array{Float64,2}, Hic::Array{Float64,2})
    MAT.matwrite(fname,{"H"=>H, "Hc"=>Hc, "Hi"=>Hi, "Hic"=> Hic, "counts"=>counts})
end

function computeJointInformation(counts::Array{Int64, 3}, nruns::Int64, trials::Array{Int64,1})

    ntrials = size(counts,1);
    ncells = size(counts,2);
    nbins = size(counts,3);
    E = zeros(Float64, nbins);
    Ec = zeros(Float64, nbins);
    Es = zeros(Float64, nruns,nbins);
    Ecs = zeros(Float64, nruns,nbins);
    for i=1:nbins
        E[i],Ec[i], Es[:,i],Ecs[:,i] = computeJointEntropy(squeeze(counts[:,:,i],3),trials,nruns)
    end
    return E,Ec,Es,Ecs
end

computeJointInformation(counts::Array{Int64, 3}, trials::Array{Int64,1})  = computeJointInformation(counts,100,trials);

function computeInformation(counts::Array{Int64,2}, trial_labels::Array{Int64,1},shuffles::Int64,word_size::Int64)
    #Compute information contained in the counts about the trial labels
    #Input:
    #   counts      : [ntrials X ]
    ntrials = size(counts,1)
    nbins = size(counts,2)
    if shuffles == 0
        E = zeros(nbins-word_size)
        Ec = zeros(nbins-word_size)
        for i=1:nbins-word_size
            E[i],Ec[i] = computeInformation(counts[:,i:i+word_size-1],trial_labels,0)
        end
    else
        E = zeros(shuffles,nbins-word_size)
        Ec = zeros(shuffles,nbins-word_size)
        tl = zeros(Int64,size(trial_labels))
        for s=1:shuffles
            tl[:] = shuffle(trial_labels)
            for i=1:nbins-word_size
                E[s,i],Ec[s,i] = computeInformation(counts[:,i:i+word_size-1],tl,0)
            end
        end
    end

    return E,Ec
end

function computeInformation(counts::Array{Int64}, trial_labels::Array{Int64,1},shuffles::Int64)
    
    ps = bincount(trial_labels)
    idx =find( ps .> 0)
    ps = ps./sum(ps)
    E = computeEntropy(counts)
    if shuffles == 0
        Ec = 0
        for i=1:length(idx)
            Ec += computeEntropy(counts[trial_labels.==idx[i]-1,:])*ps[idx[i]]
        end
    else
        Ec = zeros(shuffles)
        for k=1:shuffles
            tl = shuffle(trial_labels)
            for i=1:length(idx)
                Ec[k] += computeEntropy(counts[tl.==idx[i]-1,:])*ps[idx[i]]
            end
        end
    end
    return E,Ec
end

function computeConditionalEntropy(counts::Array{Int64},trial_labels::Array{Int64,1},shuffles::Int64)

    ps = bincount(trial_labels)
    idx =find( ps .> 0)
    ps = ps./sum(ps)
    if shuffles == 0
        Ec = 0
        for i=1:length(idx)
            Ec += computeEntropy(counts[trial_labels.==idx[i]-1,:])*ps[idx[i]]
        end
    else
        Ec = zeros(shuffles)
        for k=1:shuffles
            tl = shuffle(trial_labels)
            for i=1:length(idx)
                Ec[k] += computeEntropy(counts[tl.==idx[i]-1,:])*ps[idx[i]]
            end
        end
    end
    return Ec

end
function computeEntropy(counts::Array{Int64,1})
    P = bincount(counts)
    #normalize
    P = P./sum(P);
    E = -sum(P.*log2(P + float(P.==0)));
    return E
end

function computeEntropy(counts::Array{Int64,2})
    words = hash(counts)    
    E = computeEntropy(words)
    return E
end

function computeIndependentEntropy(counts::Array{Int64,2},nshuffles::Integer)
    word_size = size(counts,2)
    E = zeros(nshuffles)
    shuffled_counts = zeros(Int64,size(counts))
    words = zeros(Int64,size(counts,1))
    for i=1:nshuffles
        for j=1:word_size
            shuffled_counts[:,j] = shuffle(squeeze(counts[:,j],2))
        end
        words[:] = hash(shuffled_counts[:,:],3)
        E[i] = computeEntropy(words)
    end
end

function computeIndependentEntropy(counts::Array{Int64,3},nshuffles::Integer)
    word_size = size(counts,2)
    nbins = size(counts,3)
    E = zeros(nshuffles,nbins)
    shuffled_counts = zeros(Int64,size(counts))
    words = zeros(Int64,size(counts,1))
    for i=1:nshuffles
        for k=1:nbins
            for j=1:word_size
                shuffled_counts[:,j,k] = shuffle(squeeze(counts[:,j,k],(2,3)))
            end
            words[:] = hash(squeeze(shuffled_counts[:,:,k],3))
            E[i,k] = computeEntropy(words)
        end
    end
    return E

end

function computeIndependentEntropy(counts::Array{Int64,2},nshuffles::Integer)
    #Compute the entropy of the rows of the supplied matrix of counts, by
    #shuffling the columns independently, i.e. breaking up any relationship between
    #the elements in each row
    E = zeros(nshuffles)
    word_size = size(counts,2)
    shuffled_counts = zeros(Int64,size(counts))
    words = zeros(Int64,size(counts,1),1)
    for i=1:nshuffles
        for j=1:word_size
            shuffled_counts[:,j] = shuffle(counts[:,j])
        end
        words = hash(shuffled_counts) 
        E[i] = computeEntropy(words)
    end
    return E
end

function bincount(A::Array{Int64,1})
    mx = max(A)+1
    Q = zeros(Int64,mx)
    for i=1:length(A)
        Q[A[i]+1] +=1
    end
    return Q
end


function hash(A::Array{Int64,2})
    word_size = size(A,2)
    mx = max(A)+1
    cconv = mx.^[word_size-1:-1:0]
    words = A*cconv
    return words
end

function computeTemporalEntropy(counts::Array{Int64,2},word_size::Int64, trial_labels::Array{Int64,1})
    ntrials = size(counts,1)
    nbins = size(counts,2)
    unique_trials = unique(trial_labels)
    if min(unique_trials) == 0
        tl = trial_labels + 1
    else
        tl = trial_labels
    end
    ncond = length(unique_trials) + 1
    mx = max(counts) + 1
    cconv = mx.^[word_size-1:-1:0]
    N = mx^word_size
    P = zeros(Float64, N, nbins-word_size)
    Pc = zeros(Float64, N, ncond, nbins-word_size)
    for b=1:nbins-word_size
        words = counts[:,b:b+word_size-1]*cconv
        for t=1:ntrials
            P[words[t]+1,b] += 1.
            Pc[words[t]+1,tl[t],b] += 1.
        end
    end
    #normalize
    np = sum(P,1)
    P = P./broadcast(+,np,float(P.==0))
    npc = sum(Pc,1)
    Pc = Pc./broadcast(+,npc , float(Pc.==0))
    Ps = 1/ncond
    E = -sum(P.*log2(P + float((P.==0))),1)
    Ec = -sum(sum(Pc.*log2(Pc + float(Pc.==0)),1).*Ps,2)

    return squeeze(E,1),squeeze(Ec,(1,2))
end

function computeJointEntropy(counts::Array{Int64,2},trial_labels::Array{Int64,1}, nruns::Int64)
    ncells = size(counts,2);
    ntrials = size(counts,1);
    unique_counts = unique(counts);
    unique_trials = unique(trial_labels);
    ncond = length(unique_trials);
    ncounts = length(unique_counts);
    P = zeros(Float64, ncounts, ncounts);
    Pc = zeros(Float64, ncond,ncounts, ncounts);
    Ps = zeros(Float64, ncounts, ncounts,nruns);
    PCs = zeros(Float64, ncond,ncounts, ncounts,nruns);
    sidx = zeros(Int64,ntrials);
    for uc1 = 1:ncounts 
        cc1 = unique_counts[uc1]
        for uc2 = 1:ncounts
            cc2 = unique_counts[uc2]
            for t=1:ntrials
                if (counts[t,1]==cc1) & (counts[t,2]==cc2)
                    P[uc2,uc1] += 1;
                    for tt=1:ncond
                        if trial_labels[t] == unique_trials[tt]
                            Pc[tt,uc2,uc1] += 1;
                        end
                    end
                end 
            end
        end
    end
    for i=1:nruns
        sidx[:] = randperm(ntrials)
        c2 = counts[sidx,2];
        for uc1=1:ncounts
            cc1 = unique_counts[uc1]
            for uc2=1:ncounts
                cc2 = unique_counts[uc2]
                for t=1:ntrials
                    if (counts[t,1]==cc1) & (counts[t,2]==cc2)
                        Ps[uc2,uc1,i] += 1;
                        for tt=1:ncond
                            if trial_labels[t] == unique_trials[tt]
                                PCs[tt,uc2,uc1,i] += 1;
                            end
                        end
                    end 
                end
                    #Ps[uc2,uc1,i] = sum((counts[:,1].==cc1)&(c2.==cc2));
            end
        end
    end
    #normalize
    P = P./sum(P)
    Ps = Ps./sum(sum(Ps,1),2);
    Pc = Pc./sum(sum(Pc,2),3);
    PCs = PCs./sum(sum(PCs,2),3);
    E = -sum(P.*log2(P + float(P.==0)));
    Es = -squeeze(sum(sum(Ps.*log2(Ps + float(Ps.==0)),1),2),(1,2))
    Ec = -sum(sum(Pc.*log2(Pc + float(Pc.==0)),2),3);
    Ecs = -sum(sum(PCs.*log2(PCs + float(PCs.==0)),2),3);
    #stimulus
    Pq = zeros(ncond)
    nn = zeros(max(trial_labels)+1)
    for i=1:ntrials
        nn[trial_labels[i]+1] +=1;
    end
    Pq = nn[nn.>0]
    Pq = Pq./sum(Pq)
    Ec = sum(Pq.*Ec);
    Ecs = squeeze(sum(Pq[:,:,:,:].*Ecs,1),(1,2,3));

    return E,Ec,Es,Ecs
end


function computeJointSpikeCounts(spiketrains::Dict, bins::Array{FloatingPoint,1})
    
end

end #end module
