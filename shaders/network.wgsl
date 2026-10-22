// Versioned row-major tensors; src/network.rs defines persisted coefficient order.
override NETWORK_MODE: u32 = 0u;
override BRAIN_REVISION: u32 = 2u;
// Native evaluation only: 1 legacy inputs, 2 flow, 3 observations, 4 motors, 0 final.
override BRAIN_ABLATION: u32 = 0u;
@group(0) @binding(5) var<storage, read> brain_weights: array<f32>;
@group(0) @binding(6) var<storage, read_write> brain_memory: array<vec4f>;
const BRAIN_INPUT_LIMIT: f32 = 1000000.0;
fn brain_row(slot: i32, offset: u32, width: u32, row: u32, x: vec4f) -> f32 {
    let start = u32(slot) * 192u + offset + row * width;
    var v = 0.0;
    for (var j = 0u; j < 4u; j++) { v += brain_weights[start+j] * x[j]; }
    return v;
}
fn brain_input_row(slot:i32,offset:u32,row:u32,x:vec4f,extra:vec3f)->f32 {
    let width=select(4u,7u,BRAIN_REVISION>=3u);
    var v=brain_row(slot,offset,width,row,x);
    if(BRAIN_REVISION>=3u){
        let start=u32(slot)*192u+offset+row*7u+4u;
        for(var j=0u;j<3u;j++){v+=brain_weights[start+j]*extra[j];}
    }
    return v;
}
fn brain_bias(slot: i32, offset: u32, row: u32) -> f32 {
    return brain_weights[u32(slot)*192u+offset+row];
}
fn brain_state_stride()->u32 {
    if(BRAIN_REVISION>=3u){return select(1u,3u,NETWORK_MODE==2u);}
    return 2u;
}
fn brain_state_index(index:u32,stream:u32)->u32 {
    if(NETWORK_MODE==2u){return stream*(arrayLength(&brain_memory)/brain_state_stride())+index;}
    return index*brain_state_stride()+stream;
}
fn brain_heading_index(index:u32)->u32 {return brain_state_index(index,select(0u,2u,NETWORK_MODE==2u));}
fn brain_clear(index: u32) {
    if (NETWORK_MODE == 2u) {
        brain_memory[brain_state_index(index,0u)] = vec4f(0.0);
        brain_memory[brain_state_index(index,1u)] = vec4f(0.0);
    }
    if(BRAIN_REVISION>=3u && NETWORK_MODE!=0u){brain_memory[brain_heading_index(index)]=vec4f(0.0);}
}
fn brain_motor(y:vec4f)->vec4f {
    if(BRAIN_REVISION>=3u && (BRAIN_ABLATION==0u || BRAIN_ABLATION>=4u)){return 2.0*tanh(y);}
    return y;
}
fn brain_evaluate(slot: i32, input: vec4f, index: u32, stream: u32, extra:vec3f) -> vec4f {
    let x = clamp(input, vec4f(-BRAIN_INPUT_LIMIT), vec4f(BRAIN_INPUT_LIMIT));
    var result = vec4f(0.0);
    let modern=BRAIN_REVISION>=3u;
    if (NETWORK_MODE == 1u) {
        var hidden: array<f32,8>;
        for (var i=0u; i<8u; i++) {
            hidden[i]=tanh(brain_bias(slot,select(32u,56u,modern),i)+brain_input_row(slot,0u,i,x,extra));
        }
        for (var i=0u; i<4u; i++) {
            var v=brain_bias(slot,select(72u,96u,modern),i);
            for (var j=0u; j<8u; j++) {v+=brain_bias(slot,select(40u,64u,modern)+i*8u,j)*hidden[j];}
            result[i]=v;
        }
        return brain_motor(result);
    }
    let h=brain_memory[brain_state_index(index,stream)];
    var z=vec4f(0.0);
    var r=vec4f(0.0);
    for (var i=0u; i<4u; i++) {
        let zv=brain_bias(slot,select(32u,44u,modern),i)+brain_input_row(slot,0u,i,x,extra)+brain_row(slot,select(16u,28u,modern),4u,i,h);
        let rv=brain_bias(slot,select(68u,92u,modern),i)+brain_input_row(slot,select(36u,48u,modern),i,x,extra)+brain_row(slot,select(52u,76u,modern),4u,i,h);
        z[i]=1.0/(1.0+exp(-clamp(zv,-80.0,80.0)));
        r[i]=1.0/(1.0+exp(-clamp(rv,-80.0,80.0)));
    }
    var next=vec4f(0.0);
    for (var i=0u; i<4u; i++) {
        let n=tanh(brain_bias(slot,select(104u,140u,modern),i)+brain_input_row(slot,select(72u,96u,modern),i,x,extra)+brain_row(slot,select(88u,124u,modern),4u,i,r*h));
        next[i]=z[i]*h[i]+(1.0-z[i])*n;
    }
    brain_memory[brain_state_index(index,stream)]=next;
    for (var i=0u; i<4u; i++) {result[i]=brain_bias(slot,select(124u,160u,modern),i)+brain_row(slot,select(108u,144u,modern),4u,i,next);}
    return brain_motor(result);
}
