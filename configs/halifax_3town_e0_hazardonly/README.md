# halifax_3town_e0_hazardonly, E3 ablation (hazard-only)

Same network, spawns, fires, destinations, and routes as `halifax_3town_e0`. The alert schedule fires on the E0 times and areas so the fire forecast becomes visible per community, but every event carries `instruction=none`, so no evacuate order reaches the belief channel, and the RCMP door-to-door directive channel is removed.

This is the (1,0,0) corner, hazard visible with no directive. Against the E0 baseline (1,1,0) it isolates the marginal value of the evacuate order over bare hazard visibility.
