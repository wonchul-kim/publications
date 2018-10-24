function connectionSend = openSend(host, port)
d = tcpip(host, port, 'NetworkRole', 'Client');
set(d, 'OutputBufferSize', 3000000); % Set size of receiving buffer, if needed. 

%Trying to open a connection to the server.
while(1)
    try 
        fopen(d);
        break;
    catch 
        fprintf('%s \n','Cant find Server');
    end
end
connectionSend = d;
set(connectionSend,'Timeout',1000);
end