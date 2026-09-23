# halifax_3town_e0_nonotice, E3 ablation (no-notice)

Same network, spawns, fires, destinations, and routes as `halifax_3town_e0`, with the alert schedule removed entirely. There is no `alerts.json`, so `AlertSchedule.empty()` resolves `NO_ALERT` for every agent and no broadcast, directive, or door-to-door channel exists. Households become aware only through own fire perception or peer messages.

This is the (0,0,0) corner of the information triad and the E3 lower bound. It isolates what the official channels add by removing them, and it is expected to leave the downstream communities, Highland Park and the outer ring, largely unwarned until the fire reaches them.
