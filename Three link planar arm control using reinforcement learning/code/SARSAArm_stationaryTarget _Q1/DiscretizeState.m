function [ s ] = DiscretizeState( x, statelist  )

[d  s] = min(dist(statelist,x'));
