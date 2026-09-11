"""활성 prompt와 실행 registry가 현재 지원 범위에서 일치하는지 검증한다."""

import unittest

from app.prompt_loader import PromptBundleLoader
from app.subagents.prompt_loader import SubagentPromptLoader
from app.subagents.router import _create_output_model


class CurrentConfigurationTests(unittest.TestCase):
    def test_master_and_subagent_codes_match(self) -> None:
        master_codes = set(PromptBundleLoader().load().agent_codes)
        subagent_codes = set(SubagentPromptLoader().load_all())

        self.assertEqual(
            master_codes,
            {"PERFORMANCE_FEE", "RP", "QUALIFICATION"},
        )
        self.assertEqual(master_codes, subagent_codes)

    def test_active_manifests_do_not_declare_action_or_mcp_workflow(self) -> None:
        for agent_code, bundle in SubagentPromptLoader().load_all().items():
            for scenario in bundle.manifest.get("scenarios", []):
                for detail in scenario.get("details", []):
                    with self.subTest(
                        agent_code=agent_code,
                        detail_code=detail.get("code"),
                    ):
                        self.assertNotIn("interaction", detail)
                        self.assertNotIn("mcp_workflow", detail)

    def test_rag_details_extract_keywords_without_search_action(self) -> None:
        bundles = SubagentPromptLoader().load_all()
        rag_details = {
            "QUALIFICATION": {
                "NEW_MEMBER_QUALIFICATION",
                "FOREIGNER_QUALIFICATION",
                "MINOR_QUALIFICATION",
                "FAMILY_CARD_ISSUANCE_QUALIFICATION",
                "INCOME_PROOF_ACCEPTANCE_CRITERIA",
            },
            "RP": {"RP_DOCUMENT_SEARCH"},
        }
        for agent_code, detail_codes in rag_details.items():
            bundle = bundles[agent_code]
            self.assertEqual(
                bundle.manifest["parameter_definitions"]["keywords"]["value_type"],
                "string_list",
            )
            for scenario in bundle.manifest["scenarios"]:
                for detail in scenario["details"]:
                    if detail["code"] not in detail_codes:
                        continue
                    self.assertIn("rag_query", detail["parameters"])
                    self.assertIn("keywords", detail["parameters"])
                    self.assertNotIn("search_query", detail["parameters"])

    def test_rag_keywords_are_a_structured_string_array(self) -> None:
        bundle = SubagentPromptLoader().load_one(directory="rp")
        output_model = _create_output_model(bundle)

        result = output_model.model_validate(
            {
                "matches": [
                    {
                        "scenario_code": "RP_DOCUMENTS",
                        "detail_scenario_code": "RP_DOCUMENT_SEARCH",
                        "parameters": {
                            "rag_query": "RP 연결 제한 기준을 알려줘",
                            "keywords": ["RP", "연결 제한"],
                        },
                    }
                ]
            }
        )

        self.assertEqual(
            result.matches[0].parameters.keywords,
            ["RP", "연결 제한"],
        )


if __name__ == "__main__":
    unittest.main()
