# input: s1 country | id_a list_a | id_b list_b ; France rows changed between a (looser) and b (stricter)
function Fv(a,b,T){ if(a==0) return (T==0 && b==0)?1:0; return 1.25*a/(a+b+0.25*T) }
NR==1{next}
$2=="France" && $4!=$6 {
  na=($4==""?0:split($4,x,",")); nb=($6==""?0:split($6,y,",")); d=na-nb; ch++; rm+=d
  if(nb==0){ E++; Ed+=d; next }
  # removed pair(s) FP: a=nb TP, b=d FP, T=nb -> after perfect
  g=1-Fv(nb,d,nb); G+=g
  # removed pair(s) TP: a=na, T=na -> after a=nb
  l=1-Fv(nb,0,na); L+=l
  K++
}
END{
  printf "changed S1 %d, pairs removed %d, kept-group S1 %d, to-empty S1 %d (pairs %d)\n",ch,rm,K,E,Ed
  printf "sum gain if all FP G=%.1f  sum loss if all TP L=%.1f  mean g=%.3f l=%.3f  break-even q=%.3f\n",G,L,G/K,L/K,L/(G+L)
  n=split(DELTAS,D,","); split("0.5,0.9",SS,",")
  for(i=1;i<=n;i++) for(j=1;j<=2;j++){ s=SS[j]; U=D[i]*1732544
    # U = q*G-(1-q)*L + E*(q*s-(1-q)*0.95)  -> q
    q=(U+L+E*0.95)/(G+L+E*(s+0.95)); printf "  LB delta %.6f -> %.0f S1-units; s=%.1f -> implied FP rate of removed pairs q=%.3f\n",D[i],U,s,q }
}
