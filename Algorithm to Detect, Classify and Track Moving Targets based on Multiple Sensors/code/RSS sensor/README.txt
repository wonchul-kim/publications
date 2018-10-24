0. 총 3가지의 노드가 존재 
	- Agent Node: 추적해야하는 물체에 설치되는 노드로서 자신의 신호를 보르드캐스팅함
	- Receiving/Sending Node: 영역 전체에 걸쳐 뿌려지는 노드로서  Agent Node에서 보내는 신호값의 세기를 탐지하고 이 데이터를 다시 보냄
	- Computer Node: Receiving/Sending Node가 보내는 데이터를 받아 컴퓨터(matlab)으로 보냄

1. OS는 리눅스를 사용하였고, tinyOS를 설치하여야 Ubee430(노드 센서)에 대한 프로그래밍이 가능하다

2. tinyOS 설치는 참고자료 폴더에 있는 교재의 환경구축 부분을 따라하면 된다.

3. 설치를 하였다면, Ubee430을 컴퓨터에 연결하고, cmd창에서 필요한 노드의 파일에 들어간다.

 	- Agent Node의 경우 : 해당 폴더 내의 SendingMote에 들어간다.
			      cmd 창에 다음과 같은 명령어를 입력한다
			      make telosb install,(Node ID)
			*** 이때 Node ID는 1,2,3과 같이 원하는 정수를 입력하여 주면 된다.
	- Receiving/Sending Node의 경우 : 해당 폴더 내의 RssiBase에 들어간다.
			      cmd 창에 다음과 같은 명령어를 입력한다
			      make telosb install,(Node ID)
			*** 이때 Node ID는 1,2,3과 같이 원하는 정수를 입력하여 주면 된다.
	- Computer Node의 경우 : 해당 폴더 내의 RssiBase에 들어간다.
			      cmd 창에 다음과 같은 명령어를 입력한다
			      make telosb install,(Node ID)
			*** 이때 Node ID는 1,2,3과 같이 원하는 정수를 입력하여 주면 된다.


** 각각의 폴더에 존재하는 다른 폴더는 InterceptBase, java가 존재한다.
	- InterceptBase: 무선과 시리얼간의 데이터 통신을 위한 코드
	- java: tinyOS가 java의 객체를 지향하기 때문에 이를 위한 환경 셋팅이라고 할 수 있다.
	- RssiDemoMessages: 각각의 노드가 보낼 메세지 타입을 설정한다. 




** RSSdataScan.m: RSS 데이터가 들어오는지 matlab에서 확인하는 코드
	- Parameters  >>
		nr_node = 6;    % the number of target nodes
		nr_avg = 5;     % the number of RSS scanning
		port_number = 1;% the order of the serial port connected to the base node
