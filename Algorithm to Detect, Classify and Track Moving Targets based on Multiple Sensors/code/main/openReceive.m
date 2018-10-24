function connectionServer = openReceive(host, port)
t = tcpip(host, port, 'NetworkRole', 'Server');
set(t, 'InputBufferSize', 3000000); 
% Open connection to the client.
fopen(t);
fprintf('%s \n','Client Connected');
connectionServer = t;
set(connectionServer,'Timeout',.1);
end