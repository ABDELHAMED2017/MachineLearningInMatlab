% Script to compare k-means with mixture of Gaussians

% Initialize parameters
SP=.5; % Spread of each cluster (.5 large spread, .1 small spread)
K=10; D=2; T=1000; NIT=400; % Num clusters, num dimensions, num trn cases, max num it
n=100; % Num of line segments used to draw each ellipse
a=[0:2*pi/n:2*pi]'; xy=[cos(a),sin(a)];

% Create data
X=randn(D,T)*2;
tmp=floor(rand(1,T)*K)+1;
for k=1:K
    idx=find(tmp==k);
    X(:,idx)=SP*randn(D,D)*X(:,idx);
    X(:,idx)=X(:,idx)+3*repmat(randn(D,1),[1,length(idx)]);
end;

%X(:,1:20)=X(:,1:20)/1000;

% Run std EM and k-means
% mu1=randn(D,K)/5; % Random values
ii=randperm(T); mu1=X(:,ii(1:K)); % Random cases
phi1=zeros(D,D,K); for k=1:K phi1(:,:,k)=diag(10*ones(D,1)+rand(D,1)/10); end;
p1=ones(K,1)/K; lP1=zeros(K,T); LL1=zeros(NIT,1);
mu2=mu1;
phi2=zeros(D,D,K); for k=1:K phi2(:,:,k)=diag(0.001*ones(D,1)); end;
phi2=phi2*10;
p2=ones(K,1)/K; lP2=lP1; LL2=zeros(NIT,1);
figure(253); set(253,'DoubleBuffer','on');
for i=1:NIT
    % E step for std EM
    for k=1:K
        dev=X-repmat(mu1(:,k),[1,T]);
        lP(k,:)=log(p1(k)) ...
            -0.5*log(det(2*pi*phi1(:,:,k)))-0.5*sum(dev.*(phi1(:,:,k)^-1*dev),1);
    end;
    mxlP=max(lP,[],1);
    P1=exp(lP-repmat(mxlP,[K,1]));
    sumP=sum(P1,1);
    P1=P1./repmat(sumP,[K,1]);
    LL1(i)=sum(log(sumP)+mxlP);

    % E step for k-means
    for k=1:K
        dev=X-repmat(mu2(:,k),[1,T]);
        lP(k,:)=log(p2(k)) ...
            -0.5*log(det(2*pi*(phi2(:,:,k)))) ...
            -0.5*sum(dev.*((phi2(:,:,k))^-1*dev),1);
    end;
    mxlP=max(lP,[],1);
    P2=exp(lP-repmat(mxlP,[K,1]));
    sumP=sum(P2,1);
    P2=P2./repmat(sumP,[K,1]);
    LL2(i)=sum(log(sumP)+mxlP);

    % Draw current model (before M step)
    figure(253);
    subplot(1,2,1);
    plot(X(1,:),X(2,:),'r.');
    axis equal;
    hold on;
    for k=1:K
        [U,S,V]=svd(phi1(:,:,k));
        lam=2*U*S^.5;
        xyp=xy*lam'+repmat(mu1(:,k)',[size(xy,1),1]);
        h=plot([xyp(1:end-1,1),xyp(2:end,1)],[xyp(1:end-1,2),xyp(2:end,2)],'g-');
        set(h,'LineWidth',3);
    end;
    hold off;
    title([num2str(i-1) ' Iterations Standard EM: LL=' num2str(LL1(i))]);
    drawnow;
    subplot(1,2,2);
    plot(X(1,:),X(2,:),'r.');
    axis equal;
    hold on;
    plot(mu2(1,:),mu2(2,:),'go','LineWidth',3,'MarkerSize',10);
    hold off;
    title([num2str(i-1) ' Iterations k-means Clustering']);

    % M step for std EM
    p1=sum(P1,2)/T;
    for k=1:K
        nrm=sum(P1(k,:));
        mu1(:,k)=X*P1(k,:)'/nrm;
        dev=X-repmat(mu1(:,k),[1,T]);
        for d=1:D phi1(:,d,k)=dev*(P1(k,:).*dev(d,:))'/nrm; end;
        tmp=diag(phi1(:,:,k)); tmp2=tmp; tmp2(tmp2<.01)=.01; tmp2=tmp2-tmp; 
        phi1(:,:,k)=phi1(:,:,k)+diag(tmp2);
    end;

    % M step for shared-variance EM
    for k=1:K
        nrm=sum(P2(k,:));
        mu2(:,k)=X*P2(k,:)'/nrm;
    end;
%    drawnow;
    pause;
end;
