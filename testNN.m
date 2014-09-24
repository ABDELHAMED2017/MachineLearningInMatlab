% test script...

w = zeros(6,6);
w(1,3:6) = rand(1,4);
w(2,3:5) = rand(1,3);
w(3:5,6) = rand(3,1);

xx = zeros(100,2);
xx(:,1) = 1;
xx(:,2) = x_test_t;

g = zeros(100,6);
g(:,1) = xx(:,1);
g(:,2) = xx(:,2);

lr = 0.001; 

for it=1:20000
    
   for j=3:5
       xx(:,j) = g*w(:,j);
       g(:,j) = 1./(1+exp(-xx(:,j)));
   end
   
   xx(:,6) = g*w(:,6);
   g(:,6) = xx(:,6);
   
   dedx(:,6) = 2*(g(:,6)-y_test_t);
   
   for m=5:-1:3
      dedx(:,m) = dedx(:,m+1:6)*w(m,m+1:6)' .*(g(:,m).*(1-g(:,m))); 
   end
   
   del = g'*dedx;
   
   w = w - lr*del .*(w~=0);

end

figure;
plot(x_test_t, y_test_t, 'yo', x_test_t, g(:,6), 'r+');