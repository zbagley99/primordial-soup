fn add_occupancy_reference(i:u32,amount:f32) {
    let addition=i32(round(clamp(amount*DENSITY_FIXED_SCALE,0.0,f32(DENSITY_FIXED_MAX))));
    loop {
        let old=atomicLoad(&splat[i]);
        let next=min(old+addition,DENSITY_FIXED_MAX);
        let changed=atomicCompareExchangeWeak(&splat[i],old,next);
        if(changed.exchanged){break;}
    }
}
