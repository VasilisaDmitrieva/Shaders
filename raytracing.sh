#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif

uniform vec2 resolution;
uniform float time;
uniform int pointerCount;
uniform float battery;
uniform vec3 pointers[10];

struct ray
{
 vec3 O, D;
 };
struct material
{
 vec3 Ka, Kd, Ks;
 float Ph;
 float ref;
 };
struct object
{
 vec3 P;
 vec3 N;
 material M;
 float t;
 int n;
 };
material M, M1, M2, M3, M4;

mat4 RotateY( float Angle )
{
  float
   A = Angle * 3.14159 / 180.0,
   si = sin(A), co = cos(A);
   return mat4(co, 0, -si, 0,
              0, 1, 0, 0,
              si, 0, co, 0,
             0, 0, 0, 1);
}

object AllInter( const ray Ray );
vec3 PointS( const vec3 LPos, const vec3 N, const vec3 P, const ray Ray, material Mtl )
{
 vec3 lightDir = LPos - P;
 float Dist = length(lightDir);
 lightDir = lightDir / Dist;
 float att = 1.0 / (0.1 + 0.1 * Dist + 0.1 * Dist * Dist);
 vec3 R = normalize(reflect(Ray.D, N));
 vec3 diffuse = Mtl.Kd * max(0.0, dot(N, lightDir));
 vec3 specular = Mtl.Ks * pow(max(0.0, dot(R, lightDir)), Mtl.Ph);

  ray Refl;
  Refl.D = lightDir;
  Refl.O = P + vec3(0.001) * N;
  object O = AllInter(Refl);

  if (O.t > 0.0001 && O.t < 300.0)
    if (!(length(P - O.P) > length(LPos - O.P)))
      return vec3(0);

 return (diffuse + specular) * att;
}

vec3 DirS( const vec3 LDir, const vec3 N, const vec3 P, const ray Ray, material Mtl )
{
 vec3 R = normalize(reflect(Ray.D, N));

 vec3 diffuse = Mtl.Kd * max(0.0, dot(N, LDir));
 vec3 specular = Mtl.Ks * pow(max(0.0, dot(R, LDir)), Mtl.Ph);

  ray Refl;
  Refl.D = LDir;
  Refl.O = P + vec3(0.001) * N;
  object O = AllInter(Refl);

  if (O.t > 0.0001 && O.t < 300.0)
    return vec3(0);

 return diffuse + specular;
}

float sphere( const vec3 c, float R, const ray Ray )
{
 vec3 a = c - Ray.O;
 float ad = dot(a, Ray.D);
 float
   OC = dot(a, a),
   OK = ad * ad,
   h2 = R * R - OC + OK;
 if (OC < R * R)
   return ad + sqrt(h2);
  if (h2 < 0.0)
    return 400.0;
  return ad - sqrt(h2);
}

float plane( const vec3 N, const vec3 P, const ray Ray )
{
 float p = dot(N, Ray.D);
 float D = dot(N, P);
 if (abs(p) > 0.0001)
   return -(dot(N, Ray.O) - D) / p;
 return 400.0;
}

float plane1( const vec3 N, float D, const ray Ray )
{
 float p = dot(N, Ray.D);

 if (abs(p) > 0.0001)
   return -(dot(N, Ray.O) - D) / p;
 return 400.0;
}

float triangle( const vec3 P0, const vec3 P1, const vec3 P2, const ray Ray )
{
 vec3 s1 = P1 - P0, s2 = P2 - P0;
 vec3 N = cross(s1, s2);
 float D = dot(N, P0);

 float p = plane1(N, D, Ray);
 if (p > 0.0001 && p < 300.)
 {
  vec3
    T = Ray.O - P0, d = P0,
    P = cross(d, s2),
    Q = cross(s1, T);
  float t = dot(Q, s2) / dot(P, s1);

  return t;
 }

  return 306.0;
}

float box( const vec3 P1, const vec3 P2, const ray Ray )
{
 float t1, t0, n = 0.0, f = 300.0;

 for (int i = 0; i < 3; i++)
 {
   if (abs(Ray.D[i]) < 0.0001)
    if(Ray.O[i] < P1[i] || Ray.O[i] > P2[i])
      return 300.0;
    t0 = (P1[i] - Ray.O[i]) / Ray.D[i];
   t1 = (P2[i] - Ray.O[i]) / Ray.D[i];
   if (t0 > t1)
   {
    float tmp = t0;
    t0 = t1;
    t1 = tmp;
   }
   if (t0 > n)
    n = t0;
   if (t1 < f)
    f = t1;
   if (n > f || f < 0.0)
    return 300.0;
 }
 return n;
}

float quadrics( const mat4 M, const ray Ray )
{
  float a, b, c;

  a = M[0][0] * Ray.D.x * Ray.D.x +
      2.0 * M[0][1] * Ray.D.x * Ray.D.y +
      2.0 * M[0][2] * Ray.D.x * Ray.D.z +
      M[1][1] * Ray.D.y * Ray.D.y +
      M[1][2] * Ray.D.y * Ray.D.z +
      M[2][2] * Ray.D.z * Ray.D.z;
  b = 2.0 * (M[0][0] * Ray.O.x * Ray.D.x +
     M[0][1] * (Ray.D.y * Ray.O.x + Ray.D.x * Ray.O.y) +
     M[0][2] * (Ray.O.x * Ray.D.z + Ray.O.z * Ray.D.x) +
     M[0][3] * Ray.D.x + M[1][1] * Ray.O.y * Ray.D.y +
     M[2][1] * (Ray.O.y * Ray.D.z + Ray.O.z * Ray.D.y) +
     M[1][3] * Ray.D.y + M[2][2] * Ray.O.z * Ray.D.z +M[2][3] * Ray.D.z);
  c = M[0][0] * Ray.O.x * Ray.O.x +
     2.0 * M[0][1] * Ray.O.x * Ray.O.y +
     2.0 * M[0][2] * Ray.O.x * Ray.O.z +
     2.0 * M[0][3] * Ray.O.x +
     M[1][1] *Ray.O.y * Ray.O.y +
     2.0 * M[2][1] * Ray.O.y * Ray.O.z +
     2.0 * M[1][3] * Ray.O.y +
     M[2][2] * Ray.O.z * Ray.O.z +
     2.0 * M[2][3] * Ray.O.z + M[3][3];
  float t1 = (-b - sqrt(b * b - 4.0 * a * c)) / 2.0 / a,
        t2 = (-b + sqrt(b * b - 4.0 * a * c)) / 2.0 / a;

  if (t1 < 0.0)
    return t2;
  return t1;
}

float A( float x )
{
 if (x < 0.001)
   return 1.;
 return 0.0;
}

vec3 boxNormal( const vec3 P, const vec3 P1, const vec3 P2 )
{
 vec3 N;
 N = vec3(-A(P[0] - P1[0]), -A(P[1] - P1[1]), -A(P[2] - P1[2]));

 if (length(N) == 0.0)
   N = vec3(A(P[0] - P2[0]), A(P[1] - P2[1]), A(P[2] - P2[2]));
 return normalize(N);
}

object AllInter( const ray Ray )
{
 float t = 300.0;

 float t1;
 int i, j, y = -1;
  vec3 c;
 /* Intersecting camera ray with objects */

  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
    {
      c = vec3(float(i) * 3.2 -3.2, 1., float(j) * 3.2 - 3.2);
     t1 = sphere(c, 1., Ray);
     if (t1 > 0.0 && t1 < t)
     {
      t = t1;
       y = i * 3 + j;
     }
    }
  t1 = plane(vec3(0.0, 1.0, 0.0), vec3(0.0), Ray);
 if (t1 > 0.0001 && t1 < t)
 {
   t = t1;
   y = 9;
  }/*
  t1 = triangle(vec3(0.0), vec3(4.0), vec3(0, 0, 4), Ray);
 if (t1 > 0.0001 && t1 < t)
 {
   t = t1;

   P = Ray.O + Ray.D * t;
   N = vec3(0.0, 1.0, 0.0);
   Mtl = M4;
  }
*/
  object O;
  O.P = Ray.O + Ray.D * t;
  if (y >= 0 && y < 9)
  {
   c = vec3(float(y / 3) * 3.2 -3.2, 1., float(mod(float(y), 3.)) * 3.2 - 3.2);

   O.N = normalize(O.P - c);
   O.M = M4;
  }
  else if (y == 9)
  {
   O.N = vec3(0.0, 1.0, 0.0);
   O.M = M2;
  }

  O.t = t, O.n = y;

  return O;
}

vec3 Shade( const ray Cam, const object Ob )
{
 return
   //PointS(vec3(0,5,0), Ob.N, Ob.P, Cam, Ob.M);
   DirS(normalize(vec3(0,1,1)), Ob.N, Ob.P, Cam, Ob.M);
}

vec3 Refract( ray Ray, object O, int n );
vec3 Refl( ray Ray, object O, int n )
{
 ray R;
 object Obj;
 float e = 2.71, r = O.M.ref;
 vec3 Col = vec3(0);

 for (int i = 0; i < n; i++)
 {
   R.O = O.P + 0.0001 * O.N;
   R.D = normalize(reflect(Ray.D, O.N));
   Obj = AllInter(R);
   if (Obj.t < 300.0 && Obj.t > 0.0001)
     Col += (Shade(R, Obj)) * O.M.ref * pow(e, -float(i+1));
   else
   {
     Col += mix(vec3(1), vec3(0,0,1),1.-battery)/*vec3(0, battery, battery)*/ * pow(e, -float(i+1));
     break;
   }
   Ray = R;
   O = Obj;
 }
  return Col * r;
}

ray ToRay( float X, float Y )
{
  mat4 Rot = RotateY(sin(time / 7.) * 100.);
 vec3
   CamAt = vec3(0),
   CamLoc = vec3(6);
   CamLoc = (Rot * vec4(CamLoc, 1)).rgb;
  vec3
    CamDir = normalize(CamAt - CamLoc),
    CamUp = vec3(0.0,1.0,0.0),
    CamR = normalize(cross(CamDir, CamUp));
    CamUp = cross(CamDir, CamR);
 float
    W = resolution.x,
    H = resolution.y;
 float ratio_x = 0.05, ratio_y = 0.05;
  if (W > H)
   ratio_x *= W / H;
 else
   ratio_y *= H / W;
 float Wp = ratio_x * 2., Hp = ratio_y * 2.;
  /* Calculate camera ray */
 vec3 A = 0.1 * CamDir;
 vec3 B = CamR * (X + 0.5 - W / 2.0) / W * Wp;
 vec3 C = CamUp * (-Y - 0.5 + H / 2.0) / H * Hp;
 vec3 x = A + B + C;
  ray CamRay;
  CamRay.O = CamLoc + x;
  CamRay.D = normalize(x);

  return CamRay;
}

void init( void )
{
 M.Ka = vec3(0.24725,0.1995,0.0745);
 M.Kd = vec3(0.75164,0.60648,0.22648);
  M.Ks = vec3(0.628281,0.555802,0.366065);
  M.Ph = 51.2;
  M.ref = 0.9;

  M1.Ka = vec3(0.19225,0.19225,0.19225);
 M1.Kd = vec3(0.50754,0.50754,0.50754);
  M1.Ks = vec3(0.508273,0.508273,0.508273);
  M1.Ph = 51.2;
  M1.ref = 0.6;

  M2.Ka = vec3(0.1);
 M2.Kd = vec3(0.1);
  M2.Ks = vec3(0.5);
  M2.Ph = 32.0;
  M2.ref = 0.9;

  M3.Ka = vec3(0.1, 0.0, 0.4);
 M3.Kd = vec3(0.2, 0.7, 0.5);
  M3.Ks = vec3(0.4, 0.3, 0.4);
  M3.Ph = 7.0;
  M3.ref = 0.1;

  M4.Ka = vec3(0.25, 0.20725, 0.20725);
  M4.Kd = vec3(1.0, 0.829, 0.829);
  M4.Ks = vec3(0.296648);
  M4.Ph = 11.264;
  M4.ref = 0.8;
}
void main(void) {
if(gl_FragCoord.y > resolution.y * 0.96)
  gl_FragColor = vec4(0.6);
else
 {
   float
     X = gl_FragCoord.x,
     Y = gl_FragCoord.y;
   ray CamRay = ToRay(X, Y);
   vec4 Color = vec4(mix(vec3(1), vec3(0,0,1), 1.-battery), 1);
   if(time < 0.5)
      init();
   object Obj = AllInter(CamRay);
    if (Obj.t < 300.0 && Obj.t > 0.0001)
      Color = vec4(Refl(CamRay, Obj, 3) + Shade(CamRay, Obj) + Obj.M.Ka , 1.0);

    if (pointerCount != 0)
    {
     for(int i = 0; i < pointerCount; i++)
     {
        ray R = ToRay(pointers[i].x, pointers[i].y);
        object O = AllInter(R);
        if (Obj.n == O.n)
        {
         if (O.n != 9)
            Color += vec4(0, 0.5 * float(i + 1), 0.5 * float(pointerCount - i) * sin(time), 1);
          else
            Color += vec4(0.1, 0, 0, 1);
        }
      }
    }

   gl_FragColor = Color;
 }
}