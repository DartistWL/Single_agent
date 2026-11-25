"""execution classes."""
from .execution_signatures import DEVICE_ORDER, SIGNATURE_SIZE
SEMANTICS_VERSION = 'reference-household-v1'

def build_classes(env, state, action_space, masks):
    tables = {}
    context = env._ewh_service_guard_context(state)
    feasible = {x['setpoint_c'] for x in context.get('candidate_predictions', []) if x['feasible']}
    for name, choices in action_space.items():
        ids, members, representatives, signatures = ([], [], [], [])
        seen = {}
        for i, (command, valid) in enumerate(zip(choices, masks[name])):
            if not valid:
                ids.append(-1)
                continue
            key, power, exact = (('command', i), 0.0, False)
            if name == 'ewh_set_temp':
                accepted = not context['active'] or command in feasible
                if accepted:
                    power = env._ewh_heating_power(command, state['ewh_temp'], env.ewh_params['heater_max_power_kw'])
                    key, exact = (('heater', power), True)
            elif name == 'wash_machine_schedule' and (state['wash_remaining_runtime_norm'] > 0 or state['wash_finished']):
                key = ('ignored',)
            elif name in ('ev_power', 'battery_power'):
                power = command
            if key not in seen:
                seen[key] = len(members)
                members.append([])
                representatives.append(i)
                signatures.append([1.0, power / 6.6, float(exact), command / max(map(abs, choices)), state['wash_remaining_runtime_norm'] if name == 'wash_machine_schedule' else 0.0, state['wash_finished'] if name == 'wash_machine_schedule' else 0.0, state['wash_pending'] if name == 'wash_machine_schedule' else 0.0, state['wash_planned_start_norm'] if name == 'wash_machine_schedule' else 0.0])
            k = seen[key]
            ids.append(k)
            members[k].append(i)
        tables[name] = dict(version=SEMANTICS_VERSION, class_ids=ids, members=members, representatives=representatives, signatures=signatures)
    return tables
