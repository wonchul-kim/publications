0. �� 3������ ��尡 ���� 
	- Agent Node: �����ؾ��ϴ� ��ü�� ��ġ�Ǵ� ���μ� �ڽ��� ��ȣ�� ������ĳ������
	- Receiving/Sending Node: ���� ��ü�� ���� �ѷ����� ���μ�  Agent Node���� ������ ��ȣ���� ���⸦ Ž���ϰ� �� �����͸� �ٽ� ����
	- Computer Node: Receiving/Sending Node�� ������ �����͸� �޾� ��ǻ��(matlab)���� ����

1. OS�� �������� ����Ͽ���, tinyOS�� ��ġ�Ͽ��� Ubee430(��� ����)�� ���� ���α׷����� �����ϴ�

2. tinyOS ��ġ�� �����ڷ� ������ �ִ� ������ ȯ�汸�� �κ��� �����ϸ� �ȴ�.

3. ��ġ�� �Ͽ��ٸ�, Ubee430�� ��ǻ�Ϳ� �����ϰ�, cmdâ���� �ʿ��� ����� ���Ͽ� ����.

 	- Agent Node�� ��� : �ش� ���� ���� SendingMote�� ����.
			      cmd â�� ������ ���� ��ɾ �Է��Ѵ�
			      make telosb install,(Node ID)
			*** �̶� Node ID�� 1,2,3�� ���� ���ϴ� ������ �Է��Ͽ� �ָ� �ȴ�.
	- Receiving/Sending Node�� ��� : �ش� ���� ���� RssiBase�� ����.
			      cmd â�� ������ ���� ��ɾ �Է��Ѵ�
			      make telosb install,(Node ID)
			*** �̶� Node ID�� 1,2,3�� ���� ���ϴ� ������ �Է��Ͽ� �ָ� �ȴ�.
	- Computer Node�� ��� : �ش� ���� ���� RssiBase�� ����.
			      cmd â�� ������ ���� ��ɾ �Է��Ѵ�
			      make telosb install,(Node ID)
			*** �̶� Node ID�� 1,2,3�� ���� ���ϴ� ������ �Է��Ͽ� �ָ� �ȴ�.


** ������ ������ �����ϴ� �ٸ� ������ InterceptBase, java�� �����Ѵ�.
	- InterceptBase: ������ �ø����� ������ ����� ���� �ڵ�
	- java: tinyOS�� java�� ��ü�� �����ϱ� ������ �̸� ���� ȯ�� �����̶�� �� �� �ִ�.
	- RssiDemoMessages: ������ ��尡 ���� �޼��� Ÿ���� �����Ѵ�. 




** RSSdataScan.m: RSS �����Ͱ� �������� matlab���� Ȯ���ϴ� �ڵ�
	- Parameters  >>
		nr_node = 6;    % the number of target nodes
		nr_avg = 5;     % the number of RSS scanning
		port_number = 1;% the order of the serial port connected to the base node
