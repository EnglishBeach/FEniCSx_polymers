{ v_0[0] * 1000.0 * (f[0] + -1 * f[0]) } * dx(<Mesh #0>[everywhere], {})
  +  {
    -1 * (
      (
          ({ A | A_{i_{12}} = (grad(f[1]))[i_{12}] * f[0] * 0.01 * exp(-1 * f[1] / 0.13) }) 
        + ({ A | A_{i_{11}} = (grad(f[1]))[i_{11}] * -0.001 * f[0] }) 
        + ({ A | A_{i_{10}} = (grad(f[0]))[i_{10}] * f[1] * -1 * 0.01 * exp(-1 * f[1] / 0.13) }) 
        + ({ A | A_{i_8} = -0.001 * (grad(f[0]))[i_8] }) 
        + ({ A | A_{i_9} = (grad(f[0]))[i_9] * 0.001 * f[1] })
      ) 
  : (grad(v_0[0]))
    ) 
  } * dx(<Mesh #0>[everywhere], {})

  +  { v_0[1] * 1000.0 * (f[1] + -1 * f[1]) 
  } * dx(<Mesh #0>[everywhere], {})

  +  { 
    -1 * (
      (
          ({ A | A_{i_{17}} = (grad(f[0]))[i_{17}] * f[1] * 0.01 * exp(-1 * f[1] / 0.13) }) 
        + ({ A | A_{i_{16}} = (grad(f[0])) [i_{16}] * f[1] * -1 * 0.01 * exp(-1 * f[1] / 0.13) }) 
        + ({ A | A_{i_{15}} = (grad(f[1]))[i_{15}] * f[0] * -1 * 0.01 * exp(-1 * f[1] / 0.13) }) 
        + ({ A | A_{i_{13}} = (grad(f[1]))[i_{13}] * -1 * 0.01 * exp(-1 * f[1] / 0.13) }) 
        + ({ A | A_{i_{14}} = (grad(f[1]))[i_{14}] * f[0] * 0.01 * exp(-1 * f[1] / 0.13) })
      ) . (grad(v_0[1]))
    ) 
  } * dx(<Mesh #0>[everywhere], {})
  +  {
      v_0[1] * -1 
      * (((x[0]) <= (0.3)) ? (1) : (0)) 
      * (((x[0]) >= (0.1)) ? (1) : (0)) 
      * 4 * (1 + -1 * f[1] + -1 * f[0]) 
      * (
        -1 * ln((1 + -1 * f[1] + -1 * f[0]) / (1 + -1 * f[0]))
        ) ** 0.75 
  } * dx(<Mesh #0>[everywhere], {})

  +  { 0 } * ds(<Mesh #0>[everywhere], {})
  +  { 0 } * ds(<Mesh #0>[everywhere], {})


    fenics general
  +  {
 v * 100.0 * (N -N) 
} * dx

  +  {
 -((
   + ( grad(P) * E_NP * N ) 
   + ( grad(P) * -A_NM * N ) 
   + ( grad(N) * -E_NP * P ) 
   + ( -A_NM * grad(N) ) 
   + ( grad(N) * A_NM * P )
 ) : grad(v)) 
} * dx

  +  {
 u * 100.0 * (P -P) 
} * dx

  +  {
 -((
  + ( grad(N) * E_NP * P ) 
  + ( grad(N) * -B_PM * P ) 
  + ( grad(P) * -E_NP * N ) 
  + ( -B_PM * grad(P) ) 
  + ( grad(P) * B_PM * N )
  ) . grad(u)) 
} * dx

  +  {
 u * -light * REACTION 
} * dx

  +  {0 } * ds
  +  {0 } * ds

wolfram

-((
   + ( grad(P) * E_NP * N ) 
   + ( grad(P) * -A_NM * N ) 
   + ( grad(N) * -E_NP * P ) 
   + ( -A_NM * grad(N) ) 
   + ( grad(N) * A_NM * P )
 ) : grad(v)) 

-ENP n(t,x) p^(0,2)(t,x)
+ANM n(t,x) p^(0,2)(t,x)
+ENP n^(0,2)(t,x) p(t,x)
+ANM n^(0,2)(t,x)
-ANM n^(0,2)(t,x) p(t,x)
+\[CapitalGamma]n
==n^(1,0)(t,x)

-((
  + ( grad(N) * E_NP * P ) 
  + ( grad(N) * -B_PM * P ) 
  + ( grad(P) * -E_NP * N ) 
  + ( -B_PM * grad(P) ) 
  + ( grad(P) * B_PM * N )
  ) . grad(u)) 

-ENP n^(0,2)(t,x) p(t,x)
+BPM n^(0,2)(t,x) p(t,x)
+ENP n(t,x) p^(0,2)(t,x)
+BPM p^(0,2)(t,x)
-BPM n(t,x) p^(0,2)(t,x)
+\[CapitalGamma]p
+V
==p^(1,0)(t,x)

